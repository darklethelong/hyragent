from logging_function import AppLogger
from typing import Dict, Union, Optional
from transformers.agents import Tool
from mongodb import MongoDB

from transformers.agents import tool
logger = AppLogger(__name__, log_file= 'MongoDB.log')

class MongoRetrieverTool(Tool):
    name = "retriever"
    description = "Using MongoDB retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"
    
    def __init__(self):
        self.vectordb = MongoDB()
    
    def forward(self,query: str, k: int = 5):
        """
        MongoDB search

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
                    f"===== Document {str(i)} =====\n" + doc['text']
                    for i, doc in enumerate(docs)
                ][:k]
            )
        except Exception as e:
            logger.error(f"Fail combine search: {e}")
            raise