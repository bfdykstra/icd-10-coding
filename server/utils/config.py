#!/usr/bin/env python3
"""
Centralized configuration management.
"""

import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # API Configuration
    openai_api_key: str = os.getenv('OPENAI_API_KEY', '')
    summary_model: str = "gpt-5-mini"
    synthesis_model: str = "gpt-5-mini"
    
    # Database Configuration
    chroma_db_path: str = "./chroma_db"
    icd10_embedding_path: str = "./icd10_embeddings"
    patient_summary_embedding_path: str = "./patient_summary_embeddings"
    icd10_collection_name: str = "icd10_embeddings"
    patient_summary_collection_name: str = "patient_summary_embeddings"
    
    # Processing Configuration
    batch_size: int = 100
    max_concurrent: int = 10
    
    

config = Config() 