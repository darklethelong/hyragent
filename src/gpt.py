from typing import Optional, List, Dict, Any
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
pipeline,
TransformersEngine
)
from pathlib import Path
import torch
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass 
class LLMConfig:
    model_path: Path
    max_new_tokens: int
    max_length: int
    padding: str = 'max_length'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    truncation: bool = True

class LLM:
    def __init__(
        self,
        model_path: Path = Path("./phi"),
        max_new_tokens: int = 8000,
        max_length: int = 128000
    ) -> None:
        self.config = LLMConfig(
            model_path=model_path,
            max_new_tokens=max_new_tokens, 
            max_length=max_length
        )
        self.engine = self._initialize_engine()
        
    def __call__(self, prompts: List[Dict[str, str]], **kwargs) -> List[str]:
        return self.engine(prompts, **kwargs)
        
    def _initialize_engine(self) -> TransformersEngine:
        try:
            logger.info(f"Initializing LLM from {self.config.model_path}")
            engine = self._create_engine()
            self._warmup(engine)
            logger.info("LLM initialization complete")
            return engine
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise RuntimeError("LLM initialization failed") from e

    def _create_engine(self) -> TransformersEngine:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
                padding=self.config.padding
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path
            )
            
            pipe = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=self.config.max_length,
                device=self.config.device,
                max_new_tokens=self.config.max_new_tokens
            )
            
            return TransformersEngine(pipe)
            
        except Exception as e:
            logger.error(f"Failed to create engine: {str(e)}")
            raise RuntimeError("Engine creation failed") from e
            
    def _warmup(self, engine: TransformersEngine) -> None:
        try:
            logger.info("Performing LLM warmup")
            warmup_prompt = [{"role": "user", "content": "tell me a short creative joke"}]
            engine(warmup_prompt)
            logger.info("Warmup complete")
        except Exception as e:
            logger.error(f"Warmup failed: {str(e)}")
            raise RuntimeError("Model warmup failed") from e