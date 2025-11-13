# Frontend Setup Guide

## 🎨 Chainlit Web Interface

The CDER GraphRAG system now includes a beautiful web interface that displays **all 4 retrieval modes side by side** for each query.

## Quick Start

### Option 1: Using Docker (Recommended)

```powershell
# Start the frontend service
docker-compose up frontend

# Or run in background
docker-compose up -d frontend
```

Then open your browser to: **http://localhost:8000**

### Option 2: Local Installation

1. **Install Chainlit** (if not already installed):
   ```powershell
   pip install chainlit
   ```

2. **Run the frontend**:
   ```powershell
   # From project root
   chainlit run frontend/app.py
   
   # Or specify port
   chainlit run frontend/app.py --port 8000
   ```

3. **Open browser**: http://localhost:8000

## Features

### 4 Response Display
Each query shows:
1. **No-RAG (Baseline)** - Answer without retrieval
2. **Vector-Only RAG** - Semantic similarity search
3. **Graph-Only RAG** - Entity-based graph traversal
4. **Hybrid RAG** - Combined vector + graph

### For Each Response
- ✅ Full answer text
- ✅ Citations (source documents)
- ✅ Performance metrics:
  - Chunks retrieved
  - Retrieval latency
  - Generation latency
  - Total tokens
  - Confidence score

### Comparison Table
A summary table comparing all 4 modes side by side.

## Usage

1. **Start the frontend** (see Quick Start above)
2. **Enter your question** in the chat input
3. **View all 4 responses** with detailed metrics
4. **Compare performance** across different retrieval modes

## Example Questions

- "What is parallel computing?"
- "Explain MapReduce architecture"
- "What are the benefits of distributed systems?"
- "How does load balancing work?"
- "What is horizontal scaling?"

## Configuration

Edit `frontend/.chainlit/config.toml` to customize:
- App name and description
- UI theme (light/dark)
- Session timeout
- Custom CSS/JS

## Troubleshooting

### Port Already in Use
```powershell
# Use a different port
chainlit run frontend/app.py --port 8001
```

### Module Not Found
Make sure you're running from the project root or have the project in your Python path.

### Neo4j Not Connected
The system will still work! Graph-Only and Hybrid modes will show a warning, but Vector-Only and No-RAG will work perfectly.

## Development

The frontend code is in `frontend/app.py`. Key components:
- `@cl.on_chat_start` - Initializes chatbot
- `@cl.on_message` - Handles queries and displays 4 responses
- `@cl.on_chat_end` - Cleanup

## Screenshots

The interface shows:
- Welcome message with system info
- Processing indicator
- 4 separate response cards (one per mode)
- Comparison summary table

