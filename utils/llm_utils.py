#!/usr/bin/env python3
"""
Centralized LLM utilities for OpenAI API interactions.
Provides consistent client initialization, embedding generation, and error handling.
"""

import os
from typing import List, Optional
import logging
from openai import OpenAI, AsyncOpenAI
import instructor
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """Centralized OpenAI client with consistent configuration"""
    
    def __init__(self, api_key: Optional[str] = None, embedding_model: str = "text-embedding-ada-002"):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.embedding_model = embedding_model
        self._sync_client = None
        self._async_client = None
        
    @property
    def sync_client(self) -> OpenAI:
        """Lazy-loaded synchronous client"""
        if self._sync_client is None:
            self._sync_client = OpenAI(api_key=self.api_key)
        return self._sync_client
    
    @property
    def async_client(self) -> AsyncOpenAI:
        """Lazy-loaded asynchronous client"""
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client
    
    def get_instructor_client(self, model: str = "gpt-4o-mini"):
        """Get instructor-wrapped async client for structured outputs"""
        return instructor.from_openai(self.async_client), model
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding with consistent error handling"""
        try:
            response = self.sync_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            return None
    
    async def get_embedding_async(self, text: str) -> Optional[List[float]]:
        """Async version of get_embedding"""
        try:
            response = await self.async_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding for text: {text[:50]}... Error: {e}")
            return None

# Global instance for convenience
llm_client = LLMClient() 