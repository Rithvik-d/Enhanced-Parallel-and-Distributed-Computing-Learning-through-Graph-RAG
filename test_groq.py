"""Test Groq integration."""
from src.chatbot import CDERChatbot
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Testing Groq Integration")
print("=" * 60)
print()

try:
    chatbot = CDERChatbot()
    print("✓ Chatbot initialized with Groq\n")
    
    # Test query
    test_query = "What is parallel computing?"
    print(f"Test Query: {test_query}\n")
    
    # Test with graph-only mode
    print("Testing Graph-Only Retrieval with Groq...")
    result = chatbot.process_query(test_query, retrieval_mode='graph-only')
    
    print(f"\n✓ Success!")
    print(f"Answer: {result.get('answer', '')[:200]}...")
    print(f"Tokens used: {result.get('metadata', {}).get('generation', {}).get('tokens_used', {}).get('total', 0)}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

