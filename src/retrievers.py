"""
Retrieval strategies for CDER GraphRAG system.
Implements No-RAG, Vector-Only, Graph-Only, and Hybrid retrieval approaches.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set
import logging

from src.vector_db import VectorDBManager
from src.graph_db import GraphDBManager
from src.llm_interface import LLMInterface

from src.logger import setup_logger

logger = setup_logger(__name__)


class Retriever(ABC):
    """Abstract base class for all retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str) -> str:
        """
        Retrieve context for query.
        
        Args:
            query: User query text
            
        Returns:
            Retrieved context string
        """
        pass
    
    def get_metadata(self) -> Dict:
        """
        Get retrieval metadata (latency, tokens, etc.).
        
        Returns:
            Dictionary with metadata
        """
        return {}


class NoRAGRetriever(Retriever):
    """
    Baseline retriever: returns empty context.
    Used for performance comparison.
    """
    
    def __init__(self):
        """Initialize No-RAG retriever."""
        self.metadata = {
            'latency_ms': 0.0,
            'chunks_retrieved': 0,
            'tokens_retrieved': 0
        }
        logger.info("NoRAGRetriever initialized")
    
    def retrieve(self, query: str) -> str:
        """
        Return empty context (baseline).
        
        Args:
            query: User query (ignored)
            
        Returns:
            Empty string
        """
        start_time = time.time()
        context = ""
        self.metadata['latency_ms'] = (time.time() - start_time) * 1000
        return context
    
    def get_metadata(self) -> Dict:
        """Get retrieval metadata."""
        return self.metadata.copy()


class VectorRetriever(Retriever):
    """
    Vector-based retrieval using ChromaDB similarity search.
    """
    
    def __init__(
        self,
        vector_manager: VectorDBManager,
        k: int = 5
    ):
        """
        Initialize vector retriever.
        
        Args:
            vector_manager: VectorDBManager instance
            k: Number of top results to retrieve
        """
        self.vector_manager = vector_manager
        self.k = k
        self.metadata = {
            'latency_ms': 0.0,
            'chunks_retrieved': 0,
            'tokens_retrieved': 0,
            'scores': []
        }
        logger.info(f"VectorRetriever initialized: k={k}")
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve context using vector similarity search.
        
        Args:
            query: User query text
            
        Returns:
            Assembled context string
        """
        start_time = time.time()
        
        # Perform similarity search
        results = self.vector_manager.similarity_search(query, k=self.k)
        
        if not results:
            logger.warning("Vector retrieval returned no results")
            self.metadata['latency_ms'] = (time.time() - start_time) * 1000
            return ""
        
        # Assemble context
        contexts = []
        chunk_ids = []
        scores = []
        
        for chunk_id, score, text, metadata in results:
            contexts.append(f"[Source: {metadata.get('source', 'Unknown')}, "
                          f"Chapter: {metadata.get('chapter', 'N/A')}, "
                          f"Score: {score:.3f}]\n{text}")
            chunk_ids.append(chunk_id)
            scores.append(score)
        
        context = "\n\n".join(contexts)
        
        # Update metadata
        self.metadata.update({
            'latency_ms': (time.time() - start_time) * 1000,
            'chunks_retrieved': len(results),
            'tokens_retrieved': sum(len(c.split()) for c in contexts),
            'scores': scores,
            'chunk_ids': chunk_ids
        })
        
        logger.info(f"Vector retrieval: {len(results)} chunks, {self.metadata['latency_ms']:.0f}ms")
        return context
    
    def get_metadata(self) -> Dict:
        """Get retrieval metadata."""
        return self.metadata.copy()


class GraphRetriever(Retriever):
    """
    Graph-based retrieval using Neo4j entity and relationship traversal.
    """
    
    def __init__(
        self,
        graph_manager: GraphDBManager,
        vector_manager: VectorDBManager,
        llm_interface: LLMInterface,
        hops: int = 2
    ):
        """
        Initialize graph retriever.
        
        Args:
            graph_manager: GraphDBManager instance
            vector_manager: VectorDBManager instance (for chunk retrieval)
            llm_interface: LLMInterface instance (for entity extraction)
            hops: Maximum graph traversal depth
        """
        self.graph_manager = graph_manager
        self.vector_manager = vector_manager
        self.llm_interface = llm_interface
        self.hops = hops
        self.metadata = {
            'latency_ms': 0.0,
            'chunks_retrieved': 0,
            'entities_found': 0,
            'hops': hops
        }
        logger.info(f"GraphRetriever initialized: hops={hops}")
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve context using graph traversal.
        
        Args:
            query: User query text
            
        Returns:
            Assembled context string
        """
        start_time = time.time()
        
        # Extract entities from query
        entity_names = self.llm_interface.extract_entities_from_query(query)
        
        # Find entity nodes in graph
        entity_ids = self.graph_manager.entity_seed_search(query)
        
        # Also search for entities by name
        for entity_name in entity_names:
            additional_ids = self.graph_manager.entity_seed_search(entity_name)
            entity_ids.extend(additional_ids)
        
        # Remove duplicates
        entity_ids = list(dict.fromkeys(entity_ids))
        
        if not entity_ids:
            logger.warning("Graph retrieval found no seed entities")
            self.metadata['latency_ms'] = (time.time() - start_time) * 1000
            return ""
        
        # Perform graph expansion
        expansion_result = self.graph_manager.graph_expansion(entity_ids, hops=self.hops)
        chunk_ids = expansion_result['chunk_ids']
        
        if not chunk_ids:
            logger.warning("Graph expansion found no chunks")
            self.metadata['latency_ms'] = (time.time() - start_time) * 1000
            return ""
        
        # Retrieve chunk texts from vector DB
        contexts = []
        for chunk_id in chunk_ids[:20]:  # Limit to top 20 chunks
            chunk_data = self.vector_manager.get_by_id(chunk_id)
            if chunk_data:
                text = chunk_data['text']
                metadata = chunk_data.get('metadata', {})
                contexts.append(
                    f"[Source: {metadata.get('source', 'Unknown')}, "
                    f"Chapter: {metadata.get('chapter', 'N/A')}, "
                    f"Graph Path]\n{text}"
                )
        
        context = "\n\n".join(contexts)
        
        # Update metadata
        self.metadata.update({
            'latency_ms': (time.time() - start_time) * 1000,
            'chunks_retrieved': len(contexts),
            'entities_found': len(entity_ids),
            'chunk_ids': chunk_ids[:20]
        })
        
        logger.info(
            f"Graph retrieval: {len(contexts)} chunks from {len(entity_ids)} entities, "
            f"{self.metadata['latency_ms']:.0f}ms"
        )
        return context
    
    def get_metadata(self) -> Dict:
        """Get retrieval metadata."""
        return self.metadata.copy()


class HybridRetriever(Retriever):
    """
    Hybrid retrieval combining vector and graph approaches.
    Uses weighted fusion to combine results.
    """
    
    def __init__(
        self,
        vector_manager: VectorDBManager,
        graph_manager: GraphDBManager,
        llm_interface: LLMInterface,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        k: int = 5,
        hops: int = 2,
        fusion_strategy: str = "ranked_union"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_manager: VectorDBManager instance
            graph_manager: GraphDBManager instance
            llm_interface: LLMInterface instance
            vector_weight: Weight for vector results (0.0-1.0)
            graph_weight: Weight for graph results (0.0-1.0)
            k: Number of vector results
            hops: Graph traversal depth
            fusion_strategy: Fusion method ("ranked_union" or "weighted_sum")
        """
        self.vector_manager = vector_manager
        self.graph_manager = graph_manager
        self.llm_interface = llm_interface
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.k = k
        self.hops = hops
        self.fusion_strategy = fusion_strategy
        
        # Initialize sub-retrievers
        self.vector_retriever = VectorRetriever(vector_manager, k=k)
        self.graph_retriever = GraphRetriever(
            graph_manager, vector_manager, llm_interface, hops=hops
        )
        
        self.metadata = {
            'latency_ms': 0.0,
            'chunks_retrieved': 0,
            'vector_chunks': 0,
            'graph_chunks': 0
        }
        
        logger.info(
            f"HybridRetriever initialized: vector_weight={vector_weight}, "
            f"graph_weight={graph_weight}, strategy={fusion_strategy}"
        )
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve context using hybrid approach.
        
        Args:
            query: User query text
            
        Returns:
            Fused context string
        """
        start_time = time.time()
        
        # Retrieve from both sources (could be parallelized)
        vector_context = self.vector_retriever.retrieve(query)
        graph_context = self.graph_retriever.retrieve(query)
        
        # Fuse results
        if self.fusion_strategy == "ranked_union":
            context = self._ranked_union_fusion(vector_context, graph_context)
        elif self.fusion_strategy == "weighted_sum":
            context = self._weighted_sum_fusion(vector_context, graph_context)
        else:
            # Default: simple concatenation
            context = f"{vector_context}\n\n{graph_context}"
        
        # Update metadata
        vector_meta = self.vector_retriever.get_metadata()
        graph_meta = self.graph_retriever.get_metadata()
        
        self.metadata.update({
            'latency_ms': (time.time() - start_time) * 1000,
            'chunks_retrieved': vector_meta.get('chunks_retrieved', 0) + graph_meta.get('chunks_retrieved', 0),
            'vector_chunks': vector_meta.get('chunks_retrieved', 0),
            'graph_chunks': graph_meta.get('chunks_retrieved', 0),
            'vector_latency': vector_meta.get('latency_ms', 0),
            'graph_latency': graph_meta.get('latency_ms', 0)
        })
        
        logger.info(
            f"Hybrid retrieval: {self.metadata['chunks_retrieved']} total chunks "
            f"({self.metadata['vector_chunks']} vector, {self.metadata['graph_chunks']} graph), "
            f"{self.metadata['latency_ms']:.0f}ms"
        )
        
        return context
    
    def _ranked_union_fusion(
        self,
        vector_context: str,
        graph_context: str
    ) -> str:
        """
        Fuse contexts using ranked union (deduplicate and merge).
        
        Args:
            vector_context: Context from vector retrieval
            graph_context: Context from graph retrieval
            
        Returns:
            Fused context string
        """
        # Split into chunks (simplified: by double newline)
        vector_chunks = [c.strip() for c in vector_context.split("\n\n") if c.strip()]
        graph_chunks = [c.strip() for c in graph_context.split("\n\n") if c.strip()]
        
        # Deduplicate (simple: by first 100 chars)
        seen = set()
        fused_chunks = []
        
        # Add vector chunks first (higher weight)
        for chunk in vector_chunks:
            chunk_hash = hash(chunk[:100])
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                fused_chunks.append(("vector", chunk))
        
        # Add graph chunks (lower weight, skip duplicates)
        for chunk in graph_chunks:
            chunk_hash = hash(chunk[:100])
            if chunk_hash not in seen:
                seen.add(chunk_hash)
                fused_chunks.append(("graph", chunk))
        
        # Combine chunks
        context_parts = []
        for source, chunk in fused_chunks:
            context_parts.append(f"[{source.upper()}] {chunk}")
        
        return "\n\n".join(context_parts)
    
    def _weighted_sum_fusion(
        self,
        vector_context: str,
        graph_context: str
    ) -> str:
        """
        Fuse contexts using weighted combination.
        
        Args:
            vector_context: Context from vector retrieval
            graph_context: Context from graph retrieval
            
        Returns:
            Fused context string
        """
        # Simple weighted combination
        if not vector_context and not graph_context:
            return ""
        
        if not vector_context:
            return graph_context
        
        if not graph_context:
            return vector_context
        
        # Combine with weights
        combined = f"[VECTOR WEIGHT: {self.vector_weight}]\n{vector_context}\n\n"
        combined += f"[GRAPH WEIGHT: {self.graph_weight}]\n{graph_context}"
        
        return combined
    
    def get_metadata(self) -> Dict:
        """Get retrieval metadata."""
        return self.metadata.copy()

