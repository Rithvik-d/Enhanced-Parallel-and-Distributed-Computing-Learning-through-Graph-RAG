"""
Vector database management for CDER GraphRAG system.
Handles ChromaDB operations for storing and retrieving document embeddings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time

import chromadb
from chromadb.config import Settings
from langchain.schema import Document as LangChainDocument
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.logger import setup_logger

logger = setup_logger(__name__)


class VectorDBManager:
    """
    Manage ChromaDB operations for document embeddings.
    Supports both OpenAI and Sentence-Transformer embeddings.
    """
    
    def __init__(
        self,
        collection_name: str = "cder_embeddings",
        persist_dir: Optional[str] = None,
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        """
        Initialize vector database manager.
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_dir: Directory for persistent storage
            embedding_provider: "openai" or "sentence-transformers"
            embedding_model: Model name for embeddings
            api_key: OpenAI API key (required for OpenAI embeddings)
        """
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        
        # Set up persistent directory
        if persist_dir:
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.persist_dir = None
        
        # Initialize ChromaDB client
        if self.persist_dir:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False)
            )
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings(api_key)
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(
            f"VectorDBManager initialized: collection={collection_name}, "
            f"provider={embedding_provider}, model={embedding_model}"
        )
    
    def _initialize_embeddings(
        self,
        api_key: Optional[str]
    ) -> Any:
        """
        Initialize embedding model.
        
        Args:
            api_key: OpenAI API key (if using OpenAI)
            
        Returns:
            Embeddings instance
        """
        if self.embedding_provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            return OpenAIEmbeddings(
                model=self.embedding_model,
                openai_api_key=api_key
            )
        elif self.embedding_provider == "sentence-transformers":
            return HuggingFaceEmbeddings(
                model_name=self.embedding_model
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create new one.
        
        Returns:
            ChromaDB collection
        """
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "CDER GraphRAG document embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Optional metadata dictionary
            
        Returns:
            Created collection
        """
        try:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            logger.info(f"Created collection: {name}")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            raise
    
    def add_documents(
        self,
        chunks: List[LangChainDocument],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 10
    ) -> List[str]:
        """
        Add chunked documents to ChromaDB.
        
        Args:
            chunks: List of LangChain Document objects
            embeddings: Optional pre-computed embeddings
            batch_size: Batch size for processing
            
        Returns:
            List of chunk IDs added
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return []
        
        # Generate embeddings if not provided
        if embeddings is None:
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            start_time = time.time()
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            elapsed = time.time() - start_time
            logger.info(f"Generated embeddings in {elapsed:.2f}s")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embedding_list = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.metadata.get('chunk_id', f"chunk_{len(ids)}")
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            embedding_list.append(embedding)
            
            # Prepare metadata (ChromaDB requires string values)
            metadata = {
                'source': str(chunk.metadata.get('source', '')),
                'chunk_index': str(chunk.metadata.get('chunk_index', '')),
                'chapter': str(chunk.metadata.get('chapter_number', '')),
                'token_count': str(chunk.metadata.get('token_count', '')),
                'doc_id': str(chunk.metadata.get('doc_id', '')),
                'filename': str(chunk.metadata.get('filename', ''))
            }
            
            # Add entities if present
            if 'entities' in chunk.metadata:
                metadata['entities'] = ','.join(chunk.metadata['entities'])
            
            metadatas.append(metadata)
        
        # Add in batches
        chunk_ids = []
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embedding_list[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                chunk_ids.extend(batch_ids)
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch_ids)} chunks")
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                raise
        
        logger.info(f"Added {len(chunk_ids)} chunks to collection")
        return chunk_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, str, Dict[str, Any]]]:
        """
        Perform similarity search on query.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples: (chunk_id, score, text, metadata)
        """
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Prepare where clause for filtering
        where = None
        if filter_metadata:
            where = {}
            for key, value in filter_metadata.items():
                where[key] = value
        
        # Search
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where
            )
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
        
        # Format results
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                chunk_id = results['ids'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0.0
                # Convert distance to similarity score (1 - distance for cosine)
                score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
                text = results['documents'][0][i]
                metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
                
                formatted_results.append((chunk_id, score, text, metadata))
        
        logger.info(f"Similarity search returned {len(formatted_results)} results")
        return formatted_results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by chunk ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Dictionary with chunk data and metadata, or None if not found
        """
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if results['ids'] and len(results['ids']) > 0:
                return {
                    'id': results['ids'][0],
                    'text': results['documents'][0],
                    'metadata': results['metadatas'][0] if 'metadatas' in results else {},
                    'embedding': results['embeddings'][0] if 'embeddings' in results else None
                }
            return None
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None
    
    def get_context(self, chunk_ids: List[str]) -> str:
        """
        Retrieve full context from chunk IDs.
        
        Args:
            chunk_ids: List of chunk identifiers
            
        Returns:
            Assembled context string
        """
        if not chunk_ids:
            return ""
        
        try:
            results = self.collection.get(
                ids=chunk_ids,
                include=['documents', 'metadatas']
            )
            
            contexts = []
            for i, doc_id in enumerate(results['ids']):
                text = results['documents'][i]
                metadata = results['metadatas'][i] if 'metadatas' in results else {}
                
                # Format with source information
                source = metadata.get('source', 'Unknown')
                chapter = metadata.get('chapter', '')
                context_str = f"[Source: {source}, Chapter: {chapter}]\n{text}\n"
                contexts.append(context_str)
            
            return "\n\n".join(contexts)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
    
    def update_document(
        self,
        chunk_id: str,
        new_text: Optional[str] = None,
        new_embedding: Optional[List[float]] = None,
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update document in collection.
        
        Args:
            chunk_id: Chunk identifier
            new_text: Optional new text content
            new_embedding: Optional new embedding vector
            new_metadata: Optional new metadata
            
        Returns:
            True if successful
        """
        try:
            # Get existing document
            existing = self.get_by_id(chunk_id)
            if not existing:
                logger.warning(f"Chunk {chunk_id} not found for update")
                return False
            
            # Prepare update data
            update_data = {}
            if new_text:
                update_data['documents'] = [new_text]
                # Regenerate embedding if text changed
                if new_embedding is None:
                    new_embedding = self.embeddings.embed_query(new_text)
            
            if new_embedding:
                update_data['embeddings'] = [new_embedding]
            
            if new_metadata:
                update_data['metadatas'] = [new_metadata]
            
            # ChromaDB update (using upsert)
            self.collection.update(
                ids=[chunk_id],
                **update_data
            )
            
            logger.info(f"Updated chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating chunk {chunk_id}: {e}")
            return False
    
    def delete_document(self, chunk_id: str) -> bool:
        """
        Delete document from collection.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[chunk_id])
            logger.info(f"Deleted chunk {chunk_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chunk {chunk_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample to check dimensions
            sample = self.collection.peek(limit=1)
            embedding_dim = None
            if sample['embeddings'] and len(sample['embeddings']) > 0:
                embedding_dim = len(sample['embeddings'][0])
            
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'embedding_dimensions': embedding_dim,
                'embedding_provider': self.embedding_provider,
                'embedding_model': self.embedding_model,
                'persist_directory': str(self.persist_dir) if self.persist_dir else None
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'error': str(e)
            }
    
    def verify_embedding_dimensions(self) -> bool:
        """
        Verify embedding dimensions match expected values.
        
        Returns:
            True if dimensions are correct
        """
        stats = self.get_collection_stats()
        expected_dims = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'all-MiniLM-L6-v2': 384
        }
        
        expected = expected_dims.get(self.embedding_model, None)
        actual = stats.get('embedding_dimensions')
        
        if expected and actual:
            if actual != expected:
                logger.warning(
                    f"Embedding dimension mismatch: expected={expected}, actual={actual}"
                )
                return False
        
        return True

