"""
Main chatbot orchestrator for CDER GraphRAG system.
Coordinates all components and provides interactive interface.
"""

import os
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from src.config_loader import ConfigLoader, ProjectConfig
from src.doc_processor import DocumentProcessor
from src.vector_db import VectorDBManager
from src.graph_db import GraphDBManager
from src.entity_extractor import EntityExtractor
from src.llm_interface import LLMInterface
from src.retrievers import (
    NoRAGRetriever,
    VectorRetriever,
    GraphRetriever,
    HybridRetriever
)

from src.logger import setup_logger

logger = setup_logger(__name__)


class CDERChatbot:
    """
    Main chatbot orchestrator.
    Manages all components and provides query processing interface.
    """
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        env_path: str = ".env"
    ):
        """
        Initialize chatbot with all components.
        
        Args:
            config_path: Path to YAML configuration file
            env_path: Path to .env file
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path, env_path)
        self.config: ProjectConfig = self.config_loader.load()
        self.runtime_config = self.config_loader.get_runtime_config()
        
        # Validate required environment variables
        self.config_loader.validate_required_env_vars()
        
        # Initialize components
        logger.info("Initializing CDER Chatbot components...")
        
        # Document processor
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.processing.chunk_size,
            chunk_overlap=self.config.processing.chunk_overlap
        )
        
        # Vector database
        self.vector_manager = VectorDBManager(
            collection_name="cder_embeddings",
            persist_dir=self.runtime_config.get('vector_db_path', './artifacts/vector_store'),
            embedding_provider=self.config.embeddings.provider,
            embedding_model=self.config.embeddings.model,
            api_key=self.runtime_config['openai_api_key']
        )
        
        # Graph database
        self.graph_manager = GraphDBManager(
            uri=self.runtime_config['neo4j_uri'],
            user=self.runtime_config['neo4j_user'],
            password=self.runtime_config['neo4j_password'],
            database=self.runtime_config['neo4j_database']
        )
        
        # Create graph schema
        if self.config.graph_database.create_indexes:
            self.graph_manager.create_schema()
        
        # Entity extractor
        self.entity_extractor = EntityExtractor(
            llm_model=self.config.llm.model,
            api_key=self.runtime_config['openai_api_key']
        )
        
        # LLM interface
        self.llm_interface = LLMInterface(
            model=self.config.llm.model,
            api_key=self.runtime_config['openai_api_key'],
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            system_prompt=self.config.llm.system_prompt
        )
        
        # Retrievers
        self.retrievers = {
            'no-rag': NoRAGRetriever(),
            'vector-only': VectorRetriever(
                self.vector_manager,
                k=self.config.retrieval.vector_config.top_k
            ),
            'graph-only': GraphRetriever(
                self.graph_manager,
                self.vector_manager,
                self.llm_interface,
                hops=self.config.retrieval.graph_config.max_hops
            ),
            'hybrid': HybridRetriever(
                self.vector_manager,
                self.graph_manager,
                self.llm_interface,
                vector_weight=self.config.retrieval.hybrid_config.vector_weight,
                graph_weight=self.config.retrieval.hybrid_config.graph_weight,
                k=self.config.retrieval.vector_config.top_k,
                hops=self.config.retrieval.graph_config.max_hops,
                fusion_strategy=self.config.retrieval.hybrid_config.fusion_strategy
            )
        }
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        
        logger.info("CDER Chatbot initialized successfully")
    
    def process_query(
        self,
        user_query: str,
        retrieval_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Process user query through full pipeline.
        
        Args:
            user_query: User's question
            retrieval_mode: Retrieval strategy to use
            
        Returns:
            Dictionary with answer and metadata
        """
        if retrieval_mode not in self.retrievers:
            raise ValueError(f"Invalid retrieval mode: {retrieval_mode}")
        
        logger.info(f"Processing query with {retrieval_mode} retrieval")
        
        # Retrieve context
        retriever = self.retrievers[retrieval_mode]
        context = retriever.retrieve(user_query)
        retrieval_metadata = retriever.get_metadata()
        
        # Generate answer
        llm_result = self.llm_interface.generate_answer(
            user_query,
            context,
            retrieval_mode=retrieval_mode
        )
        
        # Combine results
        result = {
            'query': user_query,
            'answer': llm_result['answer'],
            'retrieval_mode': retrieval_mode,
            'metadata': {
                'retrieval': retrieval_metadata,
                'generation': {
                    'tokens_used': llm_result['tokens_used'],
                    'latency_ms': llm_result['latency_ms'],
                    'confidence': llm_result['confidence']
                },
                'citations': llm_result['citations']
            }
        }
        
        # Add to history
        self.conversation_history.append(result)
        
        return result
    
    def compare_all_retrievals(self, user_query: str) -> Dict[str, Any]:
        """
        Run query through all four retrieval approaches.
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with results from all approaches
        """
        logger.info(f"Comparing all retrieval approaches for query")
        
        results = {}
        
        for mode in ['no-rag', 'vector-only', 'graph-only', 'hybrid']:
            try:
                result = self.process_query(user_query, retrieval_mode=mode)
                results[mode] = {
                    'answer': result['answer'],
                    'metadata': result['metadata'],
                    'latency_ms': (
                        result['metadata']['retrieval'].get('latency_ms', 0) +
                        result['metadata']['generation'].get('latency_ms', 0)
                    )
                }
            except Exception as e:
                logger.error(f"Error in {mode} retrieval: {e}")
                results[mode] = {
                    'error': str(e),
                    'answer': None
                }
        
        return {
            'query': user_query,
            'results': results,
            'comparison': self._generate_comparison(results)
        }
    
    def _generate_comparison(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison metrics for all retrieval approaches.
        
        Args:
            results: Results dictionary from compare_all_retrievals
            
        Returns:
            Comparison metrics dictionary
        """
        comparison = {
            'latencies': {},
            'answer_lengths': {},
            'confidences': {},
            'chunks_retrieved': {}
        }
        
        for mode, result in results.items():
            if 'error' in result:
                continue
            
            comparison['latencies'][mode] = result.get('latency_ms', 0)
            comparison['answer_lengths'][mode] = len(result.get('answer', ''))
            comparison['confidences'][mode] = result.get('metadata', {}).get(
                'generation', {}
            ).get('confidence', 0.0)
            comparison['chunks_retrieved'][mode] = result.get('metadata', {}).get(
                'retrieval', {}
            ).get('chunks_retrieved', 0)
        
        return comparison
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Format response for display.
        
        Args:
            result: Result dictionary from process_query
            
        Returns:
            Formatted response string
        """
        answer = result['answer']
        metadata = result['metadata']
        citations = metadata.get('citations', [])
        
        # Build formatted response
        response = f"Answer ({result['retrieval_mode']}):\n{answer}\n\n"
        
        if citations:
            response += "Citations:\n"
            for citation in citations:
                response += f"  - {citation}\n"
            response += "\n"
        
        # Add metadata summary
        retrieval_meta = metadata.get('retrieval', {})
        generation_meta = metadata.get('generation', {})
        
        response += f"Metadata:\n"
        response += f"  - Retrieval latency: {retrieval_meta.get('latency_ms', 0):.0f}ms\n"
        response += f"  - Generation latency: {generation_meta.get('latency_ms', 0):.0f}ms\n"
        response += f"  - Chunks retrieved: {retrieval_meta.get('chunks_retrieved', 0)}\n"
        response += f"  - Confidence: {generation_meta.get('confidence', 0.0):.2f}\n"
        response += f"  - Tokens used: {generation_meta.get('tokens_used', {}).get('total', 0)}\n"
        
        return response
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and statistics.
        
        Returns:
            Dictionary with system status
        """
        status = {
            'vector_db': self.vector_manager.get_collection_stats(),
            'graph_db': self.graph_manager.get_graph_stats(),
            'conversation_history_length': len(self.conversation_history),
            'retrievers_available': list(self.retrievers.keys())
        }
        
        return status
    
    def interactive_chat(self) -> None:
        """
        Start interactive chat loop.
        """
        print("=" * 60)
        print("CDER GraphRAG Chatbot - Interactive Mode")
        print("=" * 60)
        print("\nCommands:")
        print("  /compare - Compare all retrieval approaches")
        print("  /explain - Show retrieval sources")
        print("  /stats - Show system statistics")
        print("  /history - Show conversation history")
        print("  /exit - Quit\n")
        
        while True:
            try:
                user_input = input("Query: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/exit':
                        print("Goodbye!")
                        break
                    elif user_input == '/compare':
                        query = input("Enter query to compare: ").strip()
                        if query:
                            results = self.compare_all_retrievals(query)
                            self._display_comparison(results)
                    elif user_input == '/stats':
                        status = self.get_system_status()
                        print(f"\nSystem Status:\n{status}\n")
                    elif user_input == '/history':
                        print(f"\nConversation History ({len(self.conversation_history)} queries)\n")
                        for i, entry in enumerate(self.conversation_history[-5:], 1):
                            print(f"{i}. {entry['query'][:50]}...")
                    else:
                        print(f"Unknown command: {user_input}")
                    continue
                
                # Process query
                result = self.process_query(user_input, retrieval_mode='hybrid')
                formatted = self.format_response(result)
                print(f"\n{formatted}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Use /exit to quit.")
            except Exception as e:
                logger.error(f"Error in interactive chat: {e}")
                print(f"Error: {e}\n")
    
    def _display_comparison(self, comparison_result: Dict[str, Any]) -> None:
        """
        Display comparison results in formatted way.
        
        Args:
            comparison_result: Result from compare_all_retrievals
        """
        print("\n" + "=" * 60)
        print("Retrieval Comparison Results")
        print("=" * 60)
        print(f"\nQuery: {comparison_result['query']}\n")
        
        for mode, result in comparison_result['results'].items():
            print(f"\n{mode.upper().replace('-', ' ')}:")
            print("-" * 40)
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Latency: {result['latency_ms']:.0f}ms")
        
        print("\n" + "=" * 60 + "\n")
    
    def close(self) -> None:
        """Close all connections and cleanup."""
        if self.graph_manager:
            self.graph_manager.close()
        logger.info("CDER Chatbot closed")

