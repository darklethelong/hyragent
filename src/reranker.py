from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional
from logging_function import AppLogger
from pydantic import BaseModel

logger = AppLogger(__name__, log_file= 'reranker.log')

class SemanticSearchResult(BaseModel):
    score: Optional[float] = None
    document: Optional[str] = None

class SemanticReranker:
    def __init__(self, model_name: str = "./gte"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings of texts

        Args:
            texts (List[str]): List input text

        Returns:
            torch.Tensor: Embedding of Text
        """
        encoded_input = self.tokenizer(
            texts,
            truncation=True,
            max_length=8192,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _mean_pooling(
                    self, 
                    model_output: torch.Tensor, 
                    attention_mask: torch.Tensor
                    ) -> torch.Tensor:
        """Using mean pooling insteal of last hidden state

        Args:
            model_output (torch.Tensor): Original output of embedding model
            attention_mask (torch.Tensor): Masked tokens tensor

        Returns:
            torch.Tensor: Mean pooling tensor
        """
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _calculate_similarity(
                            self, 
                            query_embedding: torch.Tensor, 
                            doc_embeddings: torch.Tensor
                            ) -> np.ndarray:
        """Caculate similarity between query and documents

        Args:
            query_embedding (torch.Tensor): Embedding query
            doc_embeddings (torch.Tensor): Embedding documents

        Returns:
            np.ndarray: similarity score
        """
        return torch.mm(query_embedding, doc_embeddings.transpose(0, 1)).cpu().numpy()

    def rerank(
        self, 
        query: str, 
        documents: List[str],        
    ) -> List[SemanticSearchResult]:
        """Rerank documents

        Args:
            query (str): Input query
            documents (List[str]): List documents
            final_k (int, optional): Number top documents. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: New ordered documents
        """

        # Get embeddings for query and documents
        query_embedding = self._get_embeddings([query])
        doc_embeddings = self._get_embeddings(documents)

        # Calculate semantic similarities
        similarities = self._calculate_similarity(query_embedding, doc_embeddings)[0].tolist()

        # Prepare reranked results
        return [SemanticSearchResult(score = float(similarities[idx]), document = documents[idx]) for idx in range(len(similarities))]

    def batch_rerank(
        self,
        queries: List[str],
        documents: List[str],        
        final_k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        return [
            self.rerank(query, documents, final_k)
            for query in queries
        ]


# Usage example:
if __name__ == "__main__":
    # Initialize reranker
    reranker = SemanticReranker()
    documents = [
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

    # Single query example
    query = "what is the capital of China?"
    results = reranker.rerank(
        query=query,
        documents= documents,
        final_k=10
    )
    print(results)
