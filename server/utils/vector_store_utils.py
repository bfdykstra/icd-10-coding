#!/usr/bin/env python3
"""
Centralized vector store utilities for ChromaDB operations.
Provides consistent database connection, collection management, and search operations.
"""

import chromadb
from chromadb.api.types import QueryResult
from chromadb.utils import embedding_functions

from typing import List, Dict, Optional, Any, TypedDict, Generic, TypeVar
import json
import logging

class PatientSummaryMetadata(TypedDict):
    icd_codes: List[str]
    discharge_summary: str


class ICD10Metadata(TypedDict):
    code: str
    billable: str
    href: str
    description: str

T = TypeVar('T')
# metadata type for generic use
# MetadataType = Generic[PatientSummaryMetadata, ICD10Metadata]
# class TypedQueryResult(TypedDict, Generic[T]):
#     """Typed version of ChromaDB QueryResult with specific metadata type"""
#     ids: List[List[str]]
#     embeddings: Optional[List[List[float]]]
#     documents: List[List[str]]
#     metadatas: List[List[T]]  # <- typed metadata
#     distances: List[List[float]]
#     uris: Optional[List[List[str]]]
#     data: Optional[List[List[Any]]]

class TypedQueryResult(QueryResult, Generic[T]):
    """Typed version of ChromaDB QueryResult with specific metadata type"""
    metadatas: List[List[T]]  # <- typed metadata

class VectorStore(Generic[T]):
    """Centralized ChromaDB operations"""
    
    def __init__(self, db_path: str = "./patient_summary_embeddings", collection_name: str = "patient_summary_embeddings"):
        self.db_path = db_path
        self.collection_name = collection_name
        self._client = None
        self._collection = None
    
    @property
    def client(self):
        """Lazy-loaded ChromaDB client"""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.db_path)
        return self._client
    
    @property
    def collection(self, embedding_model: str = "lokeshch19/ModernPubMedBERT"):
        """Get or create collection using specified embedding model, defaults to ModernPubMedBERT"""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(self.collection_name)
            except:
                # sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model)
                # # Collection doesn't exist, create it
                # self._collection = self.client.create_collection(
                #     name=self.collection_name,
                #     embedding_function=sentence_transformer_ef,
                #     metadata={
                #         "embedding_model": embedding_model}
                # )
                raise ValueError(f"Collection {self.collection_name} does not exist.")
        return self._collection
    
    def recreate_collection(self):
        """Delete and recreate collection for fresh start"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logging.info(f"Deleted existing collection: {self.collection_name}")
        except:
            pass  # Collection didn't exist
        
        self._collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Embeddings for mental health prompts with quality buckets and search keywords"}
        )
        return self._collection
    
    def add_documents_batch(self, documents: List[str], 
                           metadatas: List[Dict], ids: List[str], batch_size: int = 100):
        """Add documents in batches with progress tracking"""
        from tqdm import tqdm
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding to ChromaDB"):
            end_idx = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )
    def search(self, query_texts: List[str], top_k: int = 5, 
              where_clause: Optional[Dict] = None,
              where_document: Optional[Dict] = None) -> TypedQueryResult[T]:
        """Unified search interface with typed metadata return"""
        result = self.collection.query(
            query_texts=query_texts,
            n_results=top_k,
            include=['documents', 'metadatas', 'distances'],
            where=where_clause,
            where_document=where_document
        )
        # Return type is now properly typed with T
        return result  # type: ignore
    
    def get_documents(self, where_clause: Optional[Dict] = None, limit: int = None) -> Dict[str, Any]:
        """Get documents with optional filtering"""
        kwargs = {
            'include': ['documents', 'metadatas'],
            'where': where_clause
        }
        if limit:
            kwargs['limit'] = limit
        return self.collection.get(**kwargs)
    
    def build_keyword_filter(self, keywords: List[str]) -> Optional[Dict]:
        """Build ChromaDB where clause for keyword filtering"""
        if not keywords:
            return None
            
        if len(keywords) == 1:
            return {keywords[0]: True}
        else:
            return {"$or": [{keyword: True} for keyword in keywords]}
    
    def get_count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()

# Global instance for convenience
vector_store = VectorStore[PatientSummaryMetadata](db_path="./patient_summary_embeddings", collection_name="patient_summary_embeddings")