"""
Chainlit frontend for CDER GraphRAG System.
Displays all 4 retrieval modes (No-RAG, Vector-Only, Graph-Only, Hybrid) side by side.
"""

import chainlit as cl
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chatbot import CDERChatbot
from src.logger import setup_logger

logger = setup_logger(__name__)


@cl.on_chat_start
async def start():
    """Initialize chatbot when chat starts."""
    await cl.Message(
        content="Welcome to CDER GraphRAG Chatbot! 🚀\n\n"
                "I'll show you **4 different answers** for each question:\n"
                "1. **No-RAG** - Baseline without retrieval\n"
                "2. **Vector-Only RAG** - Semantic similarity search\n"
                "3. **Graph-Only RAG** - Entity-based graph traversal\n"
                "4. **Hybrid RAG** - Combined vector + graph\n\n"
                "Initializing system...",
        author="CDER Assistant"
    ).send()
    
    # Initialize chatbot
    try:
        # Check if .env exists
        env_path = project_root / ".env"
        if not env_path.exists():
            await cl.Message(
                content="⚠️ **Warning**: `.env` file not found!\n\n"
                        "Please create a `.env` file with:\n"
                        "- `OPENAI_API_KEY=your-key`\n"
                        "- `NEO4J_URI=your-uri`\n"
                        "- `NEO4J_USER=your-user`\n"
                        "- `NEO4J_PASSWORD=your-password`\n\n"
                        "The system will attempt to initialize anyway...",
                author="System"
            ).send()
        
        chatbot = CDERChatbot()
        cl.user_session.set("chatbot", chatbot)
        
        # Check which modes are available
        available_modes = list(chatbot.retrievers.keys())
        mode_status = []
        for mode in ['no-rag', 'vector-only', 'graph-only', 'hybrid']:
            if mode in available_modes:
                mode_status.append(f"✅ {mode.upper()}")
            else:
                mode_status.append(f"⚠️ {mode.upper()} (not available)")
        
        await cl.Message(
            content=f"✅ **System initialized successfully!**\n\n"
                    f"**Available Retrieval Modes:**\n" + "\n".join(mode_status) + "\n\n"
                    f"Ask me anything about Parallel and Distributed Computing!",
            author="System"
        ).send()
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to initialize chatbot: {e}", exc_info=True)
        
        # Provide helpful error messages
        help_text = ""
        if "OPENAI_API_KEY" in error_msg or "api key" in error_msg.lower():
            help_text = "\n\n**Fix**: Check your `.env` file and ensure `OPENAI_API_KEY` is set correctly."
        elif "Neo4j" in error_msg or "neo4j" in error_msg:
            help_text = "\n\n**Note**: Neo4j connection failed, but Vector-Only and No-RAG modes should still work."
        elif "ChromaDB" in error_msg or "chromadb" in error_msg.lower():
            help_text = "\n\n**Fix**: Try deleting `artifacts/vector_store` and re-indexing documents."
        
        await cl.Message(
            content=f"❌ **Failed to initialize system**\n\n"
                    f"**Error**: {error_msg}{help_text}\n\n"
                    f"**Troubleshooting:**\n"
                    f"1. Check that `.env` file exists in project root\n"
                    f"2. Verify all required environment variables are set\n"
                    f"3. Check the terminal/logs for detailed error messages\n"
                    f"4. Try refreshing the page to retry initialization",
            author="System"
        ).send()
        
        # Store error in session for debugging
        cl.user_session.set("init_error", error_msg)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages and display all 4 responses."""
    # Check for initialization retry command
    if message.content.strip().lower() in ["/retry", "/init", "/restart"]:
        await start()
        return
    
    chatbot: CDERChatbot = cl.user_session.get("chatbot")
    
    if not chatbot:
        init_error = cl.user_session.get("init_error", "Unknown error")
        await cl.Message(
            content=f"❌ **Chatbot not initialized**\n\n"
                    f"**Error**: {init_error}\n\n"
                    f"**To fix:**\n"
                    f"1. Type `/retry` to attempt re-initialization\n"
                    f"2. Or refresh the page\n"
                    f"3. Check your `.env` file and configuration",
            author="System"
        ).send()
        return
    
    query = message.content.strip()
    if not query:
        return
    
    # Show processing message
    processing_msg = await cl.Message(
        content="🔄 Processing your query through all 4 retrieval modes...",
        author="System"
    ).send()
    
    try:
        # Process through all 4 modes
        modes = ['no-rag', 'vector-only', 'graph-only', 'hybrid']
        mode_names = {
            'no-rag': '1️⃣ No-RAG (Baseline)',
            'vector-only': '2️⃣ Vector-Only RAG',
            'graph-only': '3️⃣ Graph-Only RAG',
            'hybrid': '4️⃣ Hybrid RAG'
        }
        
        results = {}
        
        # Process each mode
        for mode in modes:
            if mode not in chatbot.retrievers:
                if mode in ['graph-only', 'hybrid'] and not chatbot.graph_manager:
                    results[mode] = {
                        'answer': '⚠️ Not available (Neo4j not connected)',
                        'metadata': {},
                        'citations': []
                    }
                    continue
                else:
                    results[mode] = {
                        'answer': f'❌ Mode "{mode}" not available',
                        'metadata': {},
                        'citations': []
                    }
                    continue
            
            try:
                result = chatbot.process_query(query, retrieval_mode=mode)
                results[mode] = result
            except Exception as e:
                logger.error(f"Error processing {mode}: {e}", exc_info=True)
                results[mode] = {
                    'answer': f'❌ Error: {str(e)}',
                    'metadata': {},
                    'citations': []
                }
        
        # Remove processing message
        await processing_msg.remove()
        
        # Display all 4 responses
        for mode in modes:
            if mode not in results:
                continue
            
            result = results[mode]
            answer = result.get('answer', 'No answer generated')
            metadata = result.get('metadata', {})
            citations = result.get('citations', [])
            
            retrieval_meta = metadata.get('retrieval', {})
            generation_meta = metadata.get('generation', {})
            
            # Build response content
            content_parts = [
                f"## {mode_names.get(mode, mode.upper())}\n\n",
                f"**Answer:**\n{answer}\n\n"
            ]
            
            # Add citations if available
            if citations:
                content_parts.append("**Citations:**\n")
                for citation in citations:
                    content_parts.append(f"- {citation}\n")
                content_parts.append("\n")
            
            # Add metrics
            chunks = retrieval_meta.get('chunks_retrieved', 0)
            retrieval_latency = retrieval_meta.get('latency_ms', 0)
            generation_latency = generation_meta.get('latency_ms', 0)
            total_latency = retrieval_latency + generation_latency
            tokens = generation_meta.get('tokens_used', {}).get('total', 0)
            confidence = generation_meta.get('confidence', 0.0)
            
            content_parts.append("**Performance Metrics:**\n")
            content_parts.append(f"- Chunks Retrieved: {chunks}\n")
            content_parts.append(f"- Retrieval Latency: {retrieval_latency:.0f}ms\n")
            content_parts.append(f"- Generation Latency: {generation_latency:.0f}ms\n")
            content_parts.append(f"- Total Latency: {total_latency:.0f}ms\n")
            content_parts.append(f"- Total Tokens: {tokens}\n")
            content_parts.append(f"- Confidence: {confidence:.2f}\n")
            
            # Send message for each mode
            await cl.Message(
                content="".join(content_parts),
                author=mode_names.get(mode, mode.upper())
            ).send()
        
        # Send comparison summary
        summary_parts = [
            "## 📊 Comparison Summary\n\n",
            "| Mode | Latency (ms) | Tokens | Confidence | Chunks |\n",
            "|------|-------------|--------|------------|--------|\n"
        ]
        
        for mode in modes:
            if mode not in results:
                continue
            
            result = results[mode]
            metadata = result.get('metadata', {})
            retrieval_meta = metadata.get('retrieval', {})
            generation_meta = metadata.get('generation', {})
            
            total_latency = retrieval_meta.get('latency_ms', 0) + generation_meta.get('latency_ms', 0)
            tokens = generation_meta.get('tokens_used', {}).get('total', 0)
            confidence = generation_meta.get('confidence', 0.0)
            chunks = retrieval_meta.get('chunks_retrieved', 0)
            
            # Extract mode display name (remove emoji and number)
            mode_display = mode_names.get(mode, mode)
            if ' ' in mode_display:
                # Get text after emoji and number (e.g., "No-RAG (Baseline)" from "1️⃣ No-RAG (Baseline)")
                mode_display = ' '.join(mode_display.split(' ')[1:])
            summary_parts.append(
                f"| {mode_display} | {total_latency:.0f} | {tokens} | {confidence:.2f} | {chunks} |\n"
            )
        
        await cl.Message(
            content="".join(summary_parts),
            author="System"
        ).send()
        
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await processing_msg.remove()
        await cl.Message(
            content=f"❌ Error processing query: {str(e)}",
            author="System"
        ).send()


@cl.on_chat_end
async def on_chat_end():
    """Cleanup when chat ends."""
    chatbot: CDERChatbot = cl.user_session.get("chatbot")
    if chatbot:
        try:
            chatbot.close()
        except Exception as e:
            logger.error(f"Error closing chatbot: {e}")


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

