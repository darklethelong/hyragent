from logging_function import AppLogger
from typing import Dict, Union, Optional
from transformers.agents import Tool
from chromadb import ChromaDB, ChromaConfig
from pathlib import Path

from transformers.agents import tool
logger = AppLogger(__name__, log_file= 'ChromaDB.log')

class ChromaRetrieverTool(Tool):
    name = "retriever"
    description = "Using chromadb retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"
    
    def __init__(self, data_path : Path = "./portico_ka.json"):
        self.data_path = data_path
        self.vectordb = ChromaDB(
            config = ChromaConfig(
                data_path = self.data_path
            )
        )
    
    def forward(self,query: str, k: int = 5):
        """
        Chromadb search

        Args:
            query (str): short description text
            k (int, optional): Top k documents output. Defaults to 50.

        """
        assert isinstance(query, str), "Your search query must be a string"
        try:
            docs = self.vectordb.search_documents(
            query)
            
            return "\nRetrieved documents:\n" + "".join(
                [
                    f"===== Document {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ][:k]
            )
        except Exception as e:
            logger.error(f"Fail combine search: {e}")
            raise