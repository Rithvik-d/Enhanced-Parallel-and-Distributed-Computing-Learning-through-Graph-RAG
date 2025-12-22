"""Test graph retrieval chunk matching."""
from src.chatbot import CDERChatbot
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Testing Graph Retrieval - Chunk Matching")
print("=" * 60)
print()

# Initialize chatbot
chatbot = CDERChatbot()

if not chatbot.graph_manager:
    print("✗ Graph manager not available")
    exit(1)

# Test query
test_query = "parallel computing"

print(f"Test Query: {test_query}\n")

# Get graph retriever
graph_retriever = chatbot.retrievers.get('graph-only')
if not graph_retriever:
    print("✗ Graph retriever not available")
    exit(1)

# Test retrieval (without LLM generation)
print("Testing graph retrieval...")
try:
    context = graph_retriever.retrieve(test_query)
    
    print(f"\nRetrieved Context:")
    print(f"  Length: {len(context)} characters")
    print(f"  Preview: {context[:500]}...")
    
    metadata = graph_retriever.get_metadata()
    print(f"\nMetadata:")
    print(f"  Chunks retrieved: {metadata.get('chunks_retrieved', 0)}")
    print(f"  Entities found: {metadata.get('entities_found', 0)}")
    print(f"  Latency: {metadata.get('latency_ms', 0):.0f}ms")
    
    if len(context) > 100:
        print("\n✓ Graph retrieval is working correctly!")
        print("  Chunks are being retrieved and matched successfully.")
    else:
        print("\n⚠️ Graph retrieval returned very little context")
        print("  This might indicate chunk ID mismatch between Neo4j and ChromaDB")
        
except Exception as e:
    print(f"\n✗ Graph retrieval failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

