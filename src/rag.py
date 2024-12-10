from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Literal
from hybridsearch_tool import HybridRetrieverTool
from chromasearch_tool import ChromaRetrieverTool
from milvussearch_tool import MilvusDBRetrieverTool
from mongosearch_tool import MongoRetrieverTool
from extra_search_tools import extra_search
from gpt import LLM
from transformers.agents import ReactJsonAgent
from logging_function import AppLogger

logger = AppLogger(__name__, 'rag.log')

@dataclass
class RAGConfig:
    type_search: Literal['chromadb', 'milvusdb', 'hybrid', 'mongodb']
    path_ka_dictionary: Path = Path('portico_ka.json')
    path_system_prompt: Path = Path('/prompt/system_prompt.txt') 
    bm25_weight: float = 0.45
    model_path: Path = Path('/phi')
    max_new_tokens: int = 8000
    max_length: int = 128000
    iterations: int = 3
    verbose: int = 1

class RAG:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self.system_prompt = self._init_system_prompt()
        
        if self.config.type_search == 'hybrid':
            self.retriever_tool = self._init_hybridretriever_tool()
        elif self.config.type_search == 'chromadb':
            self.retriever_tool = self._init_chromaretriever_tool()
        elif self.config.type_search == 'milvusdb':
            self.retriever_tool = self._init_milvusretriever_tool()
        else:
            self.retriever_tool = self._init_mongoretriever_tool()
            
        self.engine = self._init_engine()
        self.rag_agent = self._init_agent()

    @lru_cache
    def _init_system_prompt(self) -> str:
        with open(self.config.path_system_prompt) as f:
            return f.read()

    def _init_hybridretriever_tool(self) -> HybridRetrieverTool:
        return HybridRetrieverTool(
            data_path=self.config.path_ka_dictionary,
            bm25_weight=self.config.bm25_weight
        )
        
    def _init_mongoretriever_tool(self) -> MongoRetrieverTool:
        return MongoRetrieverTool()
    
    def _init_chromaretriever_tool(self) -> ChromaRetrieverTool:
        return ChromaRetrieverTool()
    
    def _init_milvusretriever_tool(self) -> MilvusDBRetrieverTool:
        return MilvusDBRetrieverTool()
    
    def _init_engine(self) -> LLM:
        return LLM(
            self.config.model_path,
            self.config.max_new_tokens,
            self.config.max_length
        )

    def _init_agent(self) -> ReactJsonAgent:
        return ReactJsonAgent(
            tools=[self.retriever_tool, extra_search],
            llm_engine=self.engine,
            max_iterations=self.config.iterations,
            verbose=self.config.verbose,
            system_prompt=self.system_prompt
        )

    def solve(self, query: str) -> str:
        return self.rag_agent.run(query)