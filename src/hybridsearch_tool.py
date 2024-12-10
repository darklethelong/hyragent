from elasticsearch import ElasticSearch
from reranker import SemanticReranker
from pathlib import Path
from logging_function import AppLogger
from pydantic import BaseModel
import numpy as np
from typing import Dict, Union, Optional
from transformers.agents import Tool

from transformers.agents import tool
logger = AppLogger(__name__, log_file= 'reranker.log')

class HybridRetrieverResult(BaseModel):
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    final_score: Optional[float] = None
    document: Optional[Dict] = None

class HybridRetrieverTool(Tool):
    name = "retriever"
    description = "Using elastic search, semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"
    
    def __init__(self, data_path: Path, bm25_weight: float = 0.5, ):
        self.data_path  = Path(data_path)
        self.bm25_weight = bm25_weight
        self.elastic_search = self._init_elastic_search()
        self.semantic_rerank = self._init_semantic_rerank()
        
    def _init_elastic_search(self) -> ElasticSearch:
        """
        Initialize elastic search
        """
        return ElasticSearch(self.data_path)

    def _init_semantic_rerank(self) -> SemanticReranker:
        """
        Initialize semantic reranking
        """
        return SemanticReranker()
    
    def forward(self,query: str, k: int = 5):
        """
        Hybrid search

        Args:
            query (str): short description text
            k (int, optional): Top k documents output. Defaults to 50.

        """
        assert isinstance(query, str), "Your search query must be a string"
        try:
            # Using short description to search in BM25
            short_description = query.lower().split("Long Description Issue:")[0].strip().replace("Short Description Issue:","").strip()
            es_results = self.elastic_search.search(short_description) # List[{score, dict document}]
            bm25_scores  = [result.score for result in es_results]
            documents = [result.document for result in es_results]
            text_documents = [doc['text'] for doc in documents]
            
            # Using full query for reranking system
            rerank_results = self.semantic_rerank.rerank(query=query, documents= text_documents) # List[{score, text}]
            semantic_scores  = [result.score for result in rerank_results]
            
            # Custom final score
            combined_scores = (self.bm25_weight * np.array(bm25_scores) + 
                            (1-self.bm25_weight) * np.array(semantic_scores ))
            
            # Create ranked results
            results = [
                HybridRetrieverResult(
                    bm25_score=bm25,
                    semantic_score=sem,
                    final_score=comb,
                    document=doc
                )
                for bm25, sem, comb, doc in zip(
                    bm25_scores, semantic_scores, combined_scores, documents
                )
            ]
                
            # Sort by combined score
            results.sort(key=lambda x: x.final_score, reverse=True)
            results = results[:k]
            return "\nRetrieved documents:\n" + "".join(
            [f"===== Document ID {doc.document['document_id']} =====\n" + doc.document['text'] for doc in results]
        )
        except Exception as e:
            logger.error(f"Fail combine search: {e}")
            raise