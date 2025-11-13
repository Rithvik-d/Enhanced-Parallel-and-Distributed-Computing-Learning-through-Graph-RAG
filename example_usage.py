"""
Example usage script for CDER GraphRAG system.
Demonstrates how to use the chatbot programmatically.
"""

from src.chatbot import CDERChatbot
from src.logger import setup_logger

logger = setup_logger(__name__)


def example_single_query():
    """Example: Process a single query."""
    print("=" * 60)
    print("Example 1: Single Query")
    print("=" * 60)
    
    chatbot = CDERChatbot()
    
    query = "What is parallel computing?"
    result = chatbot.process_query(query, retrieval_mode="hybrid")
    
    print(f"\nQuery: {query}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nCitations: {result['metadata']['citations']}")
    print(f"\nConfidence: {result['metadata']['generation']['confidence']:.2f}")
    
    chatbot.close()


def example_comparison():
    """Example: Compare all retrieval approaches."""
    print("\n" + "=" * 60)
    print("Example 2: Compare All Retrieval Approaches")
    print("=" * 60)
    
    chatbot = CDERChatbot()
    
    query = "Explain MapReduce"
    comparison = chatbot.compare_all_retrievals(query)
    
    print(f"\nQuery: {query}\n")
    
    for mode, result in comparison['results'].items():
        print(f"{mode.upper().replace('-', ' ')}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Answer: {result['answer'][:150]}...")
            print(f"  Latency: {result['latency_ms']:.0f}ms")
        print()
    
    chatbot.close()


def example_system_status():
    """Example: Get system status."""
    print("\n" + "=" * 60)
    print("Example 3: System Status")
    print("=" * 60)
    
    chatbot = CDERChatbot()
    
    status = chatbot.get_system_status()
    
    print(f"\nVector DB Stats:")
    print(f"  Documents: {status['vector_db'].get('total_documents', 0)}")
    print(f"  Dimensions: {status['vector_db'].get('embedding_dimensions', 'N/A')}")
    
    print(f"\nGraph DB Stats:")
    for label, count in status['graph_db'].items():
        print(f"  {label}: {count}")
    
    print(f"\nConversation History: {status['conversation_history_length']} queries")
    
    chatbot.close()


def example_custom_retrieval():
    """Example: Use specific retrieval mode."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Retrieval Mode")
    print("=" * 60)
    
    chatbot = CDERChatbot()
    
    query = "What are the benefits of distributed systems?"
    
    # Try vector-only retrieval
    result = chatbot.process_query(query, retrieval_mode="vector-only")
    print(f"\nVector-Only Result:")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Chunks Retrieved: {result['metadata']['retrieval']['chunks_retrieved']}")
    
    chatbot.close()


if __name__ == "__main__":
    try:
        # Run examples
        example_single_query()
        example_comparison()
        example_system_status()
        example_custom_retrieval()
        
    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Set up .env with Neo4j and OpenAI credentials")
        print("  2. Placed documents in data/cder_chapters/")
        print("  3. Installed all dependencies")

