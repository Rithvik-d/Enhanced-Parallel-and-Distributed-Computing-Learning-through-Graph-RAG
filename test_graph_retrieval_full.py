"""Test full graph retrieval pipeline."""
from src.chatbot import CDERChatbot
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Testing Graph Retrieval Pipeline")
print("=" * 60)
print()

# Initialize chatbot
try:
    chatbot = CDERChatbot()
    print("✓ Chatbot initialized\n")
except Exception as e:
    print(f"✗ Failed to initialize chatbot: {e}")
    exit(1)

# Check if graph manager is available
if not chatbot.graph_manager:
    print("✗ Graph manager not available")
    exit(1)

print("✓ Graph manager available\n")

# Test query
test_query = "What is parallel computing?"

print(f"Test Query: {test_query}\n")

# Test graph-only retrieval
print("Testing Graph-Only Retrieval...")
try:
    result = chatbot.process_query(test_query, retrieval_mode='graph-only')
    
    print(f"\nResult:")
    print(f"  Answer length: {len(result.get('answer', ''))} chars")
    print(f"  Answer preview: {result.get('answer', '')[:200]}...")
    
    metadata = result.get('metadata', {})
    retrieval_meta = metadata.get('retrieval', {})
    
    print(f"\nRetrieval Metadata:")
    print(f"  Chunks retrieved: {retrieval_meta.get('chunks_retrieved', 0)}")
    print(f"  Latency: {retrieval_meta.get('latency_ms', 0):.0f}ms")
    
    citations = result.get('citations', [])
    print(f"  Citations: {len(citations)}")
    if citations:
        print(f"    Sample: {citations[0]}")
    
    if result.get('answer') and len(result.get('answer', '')) > 50:
        print("\n✓ Graph retrieval is working!")
    else:
        print("\n⚠️ Graph retrieval returned empty or very short answer")
        print("   This might indicate chunk ID mismatch between Neo4j and ChromaDB")
        
except Exception as e:
    print(f"\n✗ Graph retrieval failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

