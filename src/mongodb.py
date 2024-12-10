import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pymongo import MongoClient
from logging_function import AppLogger
from openai_embedding import OpenAIEmbedding
from datetime import datetime
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from dotenv import load_dotenv
load_dotenv(override=True)

logger = AppLogger(__name__, log_file= 'mongodb.log')

class MongoDB:
    """MongoDB client for vector similarity search."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 5000):
        """Initialize MongoDB connection and settings.
        
        Args:
            max_retries: Maximum number of connection retry attempts
            timeout: Connection timeout in milliseconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = self._connect_mongodb()
        self.database: Database = self.client[self._get_env("DATABASE_NAME")]
        self.collection: Collection = self.database[self._get_env("COLLECTION")]
        self.openai_embedding = OpenAIEmbedding()

    def _get_env(self, key: str) -> str:
        """Safely get environment variable.
        
        Args:
            key: Environment variable name
            
        Returns:
            Environment variable value
            
        Raises:
            ValueError: If environment variable is not set
        """
        value = os.environ.get(key)
        if not value:
            raise ValueError(f"Environment variable {key} not set")
        return value

    def _connect_mongodb(self) -> MongoClient:
        """Establish MongoDB connection with retry logic.
        
        Returns:
            MongoDB client instance
            
        Raises:
            ConnectionError: If connection fails after max retries
        """
        retries = 0
        while retries < self.max_retries:
            try:
                client = MongoClient(
                    host=self._get_env("DEV_MONGO"),
                    serverSelectionTimeoutMS=self.timeout
                )
                # Verify connection
                client.server_info()
                logger.info(f"Successfully connected to MongoDB at {datetime.now()}")
                return client
            except Exception as e:
                retries += 1
                logger.error(f"MongoDB connection attempt {retries} failed: {str(e)}")
                if retries == self.max_retries:
                    raise ConnectionError(f"Failed to connect to MongoDB after {self.max_retries} attempts")

    def search_documents(
        self, 
        text: str,
        path_search: str = 'emb',
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            text: Search query text
            path_search: Field name containing embeddings
            limit: Maximum number of results to return
            
        Returns:
            List of matching documents with similarity scores
        """
        if not text:
            raise ValueError("Search text cannot be empty")

        query_embedding = self.openai_embedding.get_safe_embedding(
            text=text,
            max_tokens=8192
        )

        pipeline = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "path": path_search,
                        "k": limit,
                    },
                    "returnStoredSource": True,
                }
            },
            {
                "$project": {
                    "similarityScore": {"$meta": "searchScore"},
                    "document": "$$ROOT",
                }
            }
        ]

        try:
            results = self.collection.aggregate(pipeline)
            return [doc['document'] for doc in results]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise