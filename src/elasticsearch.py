import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from processing import ProcessingData
import bm25s
import numpy as np
import Stemmer
from pathlib import Path
from pydantic import BaseModel
from logging_function import AppLogger
from dotenv import load_dotenv
load_dotenv(override=True)

logger = AppLogger(__name__, log_file= 'elasticsearch.log')

@dataclass
class SearchConfig:
    data_path: Path
    bm25_index_path: Path = Path('bm25_index')
    language: str = "english"
    top_k: int = 50

class ElasticSearchResult(BaseModel):
    score: Optional[float] = None
    document: Optional[Dict] = None

class ElasticSearch:
    def __init__(self, data_path: Path, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig(data_path=data_path)
        self.stemmer = Stemmer.Stemmer(self.config.language)
        self.corpus, self.text_corpus = self._init_corpus()
        self.retriever = self._load_or_create_index()

    def _init_corpus(self) -> Tuple[List[Dict], List[str]]:
        """Initialize and process document corpus"""
        try:
            processing = ProcessingData(self.config.data_path)
            corpus, _ = processing.processed_documents()
            text_corpus = [doc['text'] for doc in corpus]
            return corpus, text_corpus
        except Exception as e:
            logger.error(f"Failed to initialize corpus: {e}")
            raise

    def _create_new_index(self) -> bm25s.BM25:
        """Create new BM25 index"""
        try:
            logger.info("Creating new BM25 index")
            corpus_tokens = bm25s.tokenize(
                self.text_corpus, 
                stopwords="en", 
                stemmer=self.stemmer
            )
            retriever = bm25s.BM25(method="atire", idf_method="lucene")
            retriever.index(corpus_tokens)
            retriever.save(str(self.config.bm25_index_path))
            return retriever
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise

    def _load_existing_index(self) -> bm25s.BM25:
        """Load existing BM25 index"""
        try:
            logger.info("Loading existing BM25 index")
            return bm25s.BM25.load(
                str(self.config.bm25_index_path), 
                load_corpus=True
            )
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise

    def _load_or_create_index(self) -> bm25s.BM25:
        """Load existing or create new BM25 index"""
        return (self._load_existing_index() 
                if self.config.bm25_index_path.exists() 
                else self._create_new_index())

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize search scores to [0,1] range"""
        scores = list(scores)[0]
        return (scores - scores.min()) / (scores.max() - scores.min())

    def _create_search_results(
        self, 
        normalized_scores: np.ndarray, 
        results: List[int]
    ) -> List[ElasticSearchResult]:
        """Create search results with normalized scores"""
        return [
            ElasticSearchResult(
                score=score,
                document=self.corpus[idx]
            )
            for score, idx in zip(normalized_scores, list(results)[0])
        ]

    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[ElasticSearchResult]:
        """
        Search documents using BM25 retrieval
        
        Args:
            query: Search query string
            top_k: Number of top results to return (defaults to config value)
            
        Returns:
            List of search results with scores and documents
        """
        try:
            top_k = top_k or self.config.top_k
            query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)
            results, scores = self.retriever.retrieve(query_tokens, k=top_k)
            
            normalized_scores = self._normalize_scores(scores)
            return self._create_search_results(normalized_scores, results)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise