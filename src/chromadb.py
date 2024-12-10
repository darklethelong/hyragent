from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from processing import ProcessingData
from pathlib import Path
from logging_function import AppLogger
from langchain.vectorstores.chroma import Chroma # Importing Chroma vector store from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_huggingface import HuggingFaceEmbeddings
import shutil # Importing shutil module for high-level file operations

logger = AppLogger(__name__, log_file= 'chromadb.log')

from dotenv import load_dotenv
load_dotenv(override=True)

@dataclass
class ChromaConfig:
    data_path: Path
    database_path: Path = "./chroma_index"
    embedding_model_name: str = "./gte"
    top_k: int = 5
    model_kwargs: dict = None
    
    def __post_init__(self):
        self.model_kwargs = self.model_kwargs or {'trust_remote_code': True}
        self.data_path = Path(self.data_path)
        self.database_path = Path(self.database_path)

class ChromaDB:
    """Vector database implementation using Chroma"""
    
    def __init__(self, config: ChromaConfig):
        """
        Initialize ChromaDB with configuration
        
        Args:
            config: ChromaConfig object containing database settings
        """
        self.config = config
        self._setup_database()
        
    def _setup_database(self) -> None:
        """Set up the vector database"""
        try:
            self._clear_existing_db()
            self.documents = self._load_documents()
            self.vectordb = self._initialize_vectordb()
            self.retriever = self._setup_retriever()
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise

    def _clear_existing_db(self) -> None:
        """Remove existing database if present"""
        try:
            if self.config.database_path.exists():
                logger.info("Clearing existing database")
                shutil.rmtree(self.config.database_path)
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            raise

    def _load_documents(self) -> List[Document]:
        """Load and process documents"""
        try:
            logger.info("Loading documents")
            processing = ProcessingData(self.config.data_path)
            return processing.processed_documents()[1]
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise

    def _get_embedding_model(self) -> HuggingFaceEmbeddings:
        """Initialize embedding model"""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_name,
                model_kwargs=self.config.model_kwargs
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def _initialize_vectordb(self) -> Chroma:
        """Initialize Chroma vector database"""
        try:
            logger.info("Initializing vector database")
            embedding_model = self._get_embedding_model()
            return Chroma.from_documents(
                documents=self.documents,
                embedding=embedding_model,
                persist_directory=str(self.config.database_path)
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise

    def _setup_retriever(self):
        """Set up document retriever"""
        try:
            return self.vectordb.as_retriever(
                search_kwargs={"k": self.config.top_k}
            )
        except Exception as e:
            logger.error(f"Failed to setup retriever: {e}")
            raise

    def search_documents(self, query: str) -> List[Document]:
        """
        Search documents using the query
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        try:
            logger.info(f"Searching documents with query: {query}")
            return self.retriever.get_relevant_documents(query=query)
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def refresh_database(self) -> None:
        """Refresh the vector database"""
        try:
            logger.info("Refreshing database")
            self._setup_database()
        except Exception as e:
            logger.error(f"Failed to refresh database: {e}")
            raise