# CDER GraphRAG Frontend

Chainlit-based web interface for the CDER GraphRAG system that displays all 4 retrieval modes side by side.

## Features

- **4 Response Display**: Shows No-RAG, Vector-Only, Graph-Only, and Hybrid responses simultaneously
- **Performance Metrics**: Displays latency, tokens, confidence, and chunks for each mode
- **Comparison Table**: Side-by-side comparison of all modes
- **Citations**: Shows source citations for each answer
- **Real-time Processing**: Streams responses as they're generated

## Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install chainlit
   ```

2. **Run the frontend**:
   ```bash
   # From project root
   chainlit run frontend/app.py
   
   # Or from frontend directory
   cd frontend
   chainlit run app.py
   ```

3. **Access the UI**:
   - Open your browser to `http://localhost:8000`

## Using Docker

```powershell
# Build and run
docker-compose run --rm cder-graphrag chainlit run frontend/app.py --port 8000

# Or add to docker-compose.yml for persistent service
```

## Configuration

Edit `.chainlit/config.toml` to customize:
- App name and description
- UI theme (light/dark)
- Session timeout
- Custom CSS/JS

## Usage

1. Start the frontend
2. Enter your question in the chat
3. View all 4 responses with metrics and citations
4. Compare performance across modes

