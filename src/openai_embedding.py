from typing import List, Iterator, Tuple, Union
import os
import openai
import tiktoken
import numpy as np
from itertools import islice
from tenacity import (
    retry,
    wait_random_exponential,
    stop_after_attempt,
    retry_if_not_exception_type
)
import logging

logger = logging.getLogger(__name__)

class OpenAIEmbedding:
    """Class for generating embeddings using Azure OpenAI API."""
    
    MAX_TOKENS: int = 8191
    DIMENSIONS: int = 1536
    DEFAULT_MODEL: str = "non-prod-text-embedding-ada-002"
    DEFAULT_ENCODING: str = "cl100k_base"
    
    def __init__(self):
        """Initialize Azure OpenAI client."""
        self.client = self._init_client()

    def _init_client(self) -> openai.AzureOpenAI:
        """Initialize and return Azure OpenAI client.
        
        Raises:
            ValueError: If required environment variables are not set
        """
        required_env_vars = [
            "GPT35_16K_OPENAI_API_BASE",
            "GPT35_16K_OPENAI_API_KEY",
            "GPT35_16K_OPENAI_API_VERSION"
        ]
        
        for var in required_env_vars:
            if not os.getenv(var):
                raise ValueError(f"Missing required environment variable: {var}")
                
        return openai.AzureOpenAI(
            azure_endpoint=os.getenv("GPT35_16K_OPENAI_API_BASE"),
            api_key=os.getenv("GPT35_16K_OPENAI_API_KEY"),
            api_version=os.getenv("GPT35_16K_OPENAI_API_VERSION")
        )

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(openai.BadRequestError)
    )
    def get_embedding(self, text_or_tokens: Union[str, List[int]], model: str = DEFAULT_MODEL) -> List[float]:
        """Get embedding from OpenAI API.

        Args:
            text_or_tokens: Input text or tokens
            model: Model name for embedding generation

        Returns:
            List of embedding values
        """
        try:
            response = self.client.embeddings.create(
                input=text_or_tokens,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

    def batched(self, iterable: Iterator, n: int) -> Iterator[Tuple]:
        """Batch data into tuples of length n.

        Args:
            iterable: Input iterator
            n: Batch size

        Yields:
            Batches of the input iterator
            
        Raises:
            ValueError: If batch size is less than 1
        """
        if n < 1:
            raise ValueError('Batch size must be at least 1')
        it = iter(iterable)
        while (batch := tuple(islice(it, n))):
            yield batch

    def chunked_tokens(
        self,
        text: str,
        encoding_name: str,
        chunk_length: int
    ) -> Iterator[List[int]]:
        """Split text into token chunks.

        Args:
            text: Input text
            encoding_name: Name of the token encoding
            chunk_length: Maximum length of each chunk

        Yields:
            Chunks of tokens
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        yield from self.batched(tokens, chunk_length)

    def get_safe_embedding(
        self,
        text: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = MAX_TOKENS,
        encoding_name: str = DEFAULT_ENCODING,
        average: bool = True
    ) -> List[float]:
        """Generate embeddings for long text by chunking.

        Args:
            text: Input text
            model: Model name
            max_tokens: Maximum tokens per chunk
            encoding_name: Name of the token encoding
            average: Whether to average chunk embeddings

        Returns:
            Combined embedding vector
        """
        if not text.strip():
            return []

        chunk_embeddings = []
        chunk_lens = []

        for chunk in self.chunked_tokens(text, encoding_name, max_tokens):
            embedding = self.get_embedding(chunk, model)
            if embedding:
                chunk_embeddings.append(embedding)
                chunk_lens.append(len(chunk))

        if not chunk_embeddings:
            logger.warning(f"No embeddings generated for text: {text[:100]}...")
            return []

        if average and len(chunk_embeddings) > 1:
            try:
                averaged = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
                normalized = averaged / np.linalg.norm(averaged)
                return normalized.tolist()
            except Exception as e:
                logger.error(f"Error averaging embeddings: {str(e)}")
                return chunk_embeddings[0]

        return chunk_embeddings[0]