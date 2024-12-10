import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Union, Tuple, Optional
from logging_function import AppLogger
from langchain_text_splitters import CharacterTextSplitter
from transformers import AutoTokenizer
from langchain.schema import Document
from dataclasses import dataclass

logger = AppLogger(__name__, log_file='processing.log')

@dataclass
class DocumentData:
    document_id: str
    text: str

class ProcessingData:
    """Class for processing and chunking text documents."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        embedding_model: str = './gte',
        custom_embedding: bool = False
    ) -> None:
        """Initialize document processor.
        
        Args:
            data_path: Path to JSON data file
            embedding_model: Path to embedding model
            custom_embedding: Whether to use custom embeddings
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        self.embedding_model = Path(embedding_model)
        self.custom_embedding = custom_embedding
        self.data_path = Path(data_path)
        self.data = self._load_data()
        self.tokenizer = None
        
        if custom_embedding:
            self.tokenizer = self._load_embedding_tokenizer()

    def _load_embedding_tokenizer(self) -> AutoTokenizer:
        """Load the embedding tokenizer.
        
        Returns:
            Loaded tokenizer
            
        Raises:
            RuntimeError: If tokenizer fails to load
        """
        try:
            logger.info(f"Loading tokenizer from {self.embedding_model}")
            return AutoTokenizer.from_pretrained(str(self.embedding_model))
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise RuntimeError("Failed to load embedding tokenizer")

    def _load_data(self) -> List[Dict]:
        """Load and parse JSON data file.
        
        Returns:
            List of document dictionaries
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            JSONDecodeError: If JSON parsing fails
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        try:
            logger.info(f"Loading data from {self.data_path}")
            with open(self.data_path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            raise
            
    def _create_original_documents(self) -> List[DocumentData]:
        """Create list of original documents.
        
        Returns:
            List of DocumentData objects
        """
        return [
            DocumentData(
                document_id=doc['document_id'],
                text=doc['text']
            )
            for doc in self.data
        ]

    def _create_langchain_documents(self) -> List[Document]:
        """Create Langchain document objects.
        
        Returns:
            List of Langchain Documents
        """
        return [
            Document(
                page_content=doc['text'],
                metadata={"document_id": doc["document_id"]}
            )
            for doc in self.data
        ]
               
    def processed_documents(self) -> Tuple[List[Dict], Optional[List[Document]]]:
        """Process and chunk documents.
        
        Returns:
            Tuple containing:
                - List of original documents
                - List of chunked Langchain documents (if custom_embedding=True)
                
        Raises:
            RuntimeError: If document processing fails
        """
        try:
            logger.info("Processing documents")
            original_documents = [
                {
                    'document_id': d.document_id,
                    'text': d.text
                }
                for d in self._create_original_documents()
            ]
            
            logger.info(f"Processed {len(original_documents)} documents")
            
            if not self.custom_embedding:
                return original_documents, None
                
            text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
                self.tokenizer,
                chunk_size=8000,
                chunk_overlap=100
            )
            
            langchain_documents = text_splitter.split_documents(
                self._create_langchain_documents()
            )
            
            logger.info(f"Created {len(langchain_documents)} document chunks")
            return original_documents, langchain_documents
            
        except Exception as e:
            logger.error(f"Document processing failed: {str(e)}")
            raise RuntimeError("Failed to process documents") from e