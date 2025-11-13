"""
Compare all 4 retrieval modes and display all answers side by side.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.chatbot import CDERChatbot

def main():
    """Run query through all 4 retrieval modes and display results."""
    
    if len(sys.argv) < 2:
        query = input("Enter your question: ").strip()
    else:
        query = " ".join(sys.argv[1:])
    
    if not query:
        print("Error: No query provided")
        return
    
    print("=" * 80)
    print("CDER GraphRAG - All 4 Retrieval Modes Comparison")
    print("=" * 80)
    print(f"\nQuery: {query}\n")
    print("Processing through all retrieval modes...\n")
    
    # Initialize chatbot
    chatbot = CDERChatbot()
    
    # Process through all modes
    results = {}
    modes = ['no-rag', 'vector-only', 'graph-only', 'hybrid']
    
    for mode in modes:
        if mode not in chatbot.retrievers:
            print(f"[SKIP] {mode.upper().replace('-', ' ')}: Not available (Neo4j not connected)")
            results[mode] = None
            continue
        
        print(f"Processing {mode.upper().replace('-', ' ')}...")
        try:
            result = chatbot.process_query(query, retrieval_mode=mode)
            results[mode] = result
        except Exception as e:
            print(f"[ERROR] {mode}: {e}")
            results[mode] = None
    
    # Display all results
    print("\n" + "=" * 80)
    print("RESULTS - All 4 Retrieval Modes")
    print("=" * 80)
    
    mode_names = {
        'no-rag': '1. NO-RAG (Baseline)',
        'vector-only': '2. VECTOR-ONLY RAG',
        'graph-only': '3. GRAPH-ONLY RAG',
        'hybrid': '4. HYBRID RAG (Vector + Graph)'
    }
    
    for mode in modes:
        print("\n" + "-" * 80)
        print(mode_names.get(mode, mode.upper()))
        print("-" * 80)
        
        if results[mode] is None:
            print("[NOT AVAILABLE]")
            continue
        
        answer = results[mode]['answer']
        metadata = results[mode]['metadata']
        retrieval_meta = metadata.get('retrieval', {})
        generation_meta = metadata.get('generation', {})
        
        print(f"\nAnswer:\n{answer}\n")
        
        if metadata.get('citations'):
            print("Citations:")
            for citation in metadata['citations']:
                print(f"  - {citation}")
            print()
        
        print("Performance Metrics:")
        print(f"  - Chunks Retrieved: {retrieval_meta.get('chunks_retrieved', 0)}")
        print(f"  - Retrieval Latency: {retrieval_meta.get('latency_ms', 0):.0f}ms")
        print(f"  - Generation Latency: {generation_meta.get('latency_ms', 0):.0f}ms")
        print(f"  - Total Latency: {retrieval_meta.get('latency_ms', 0) + generation_meta.get('latency_ms', 0):.0f}ms")
        print(f"  - Total Tokens: {generation_meta.get('tokens_used', {}).get('total', 0)}")
        print(f"  - Confidence: {generation_meta.get('confidence', 0.0):.2f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n{'Mode':<20} {'Latency (ms)':<15} {'Tokens':<10} {'Confidence':<12} {'Chunks':<10}")
    print("-" * 80)
    
    for mode in modes:
        if results[mode] is None:
            print(f"{mode_names.get(mode, mode):<20} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<10}")
            continue
        
        metadata = results[mode]['metadata']
        retrieval_meta = metadata.get('retrieval', {})
        generation_meta = metadata.get('generation', {})
        
        total_latency = retrieval_meta.get('latency_ms', 0) + generation_meta.get('latency_ms', 0)
        tokens = generation_meta.get('tokens_used', {}).get('total', 0)
        confidence = generation_meta.get('confidence', 0.0)
        chunks = retrieval_meta.get('chunks_retrieved', 0)
        
        mode_display = mode_names.get(mode, mode).split('. ')[1] if '. ' in mode_names.get(mode, mode) else mode
        print(f"{mode_display:<20} {total_latency:<15.0f} {tokens:<10} {confidence:<12.2f} {chunks:<10}")
    
    print("\n" + "=" * 80)
    
    chatbot.close()

if __name__ == "__main__":
    main()

