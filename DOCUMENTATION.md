# CDER GraphRAG System - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation & Setup](#installation--setup)
4. [Configuration](#configuration)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Docker Deployment](#docker-deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

---

## Overview

The CDER GraphRAG System is a hybrid Retrieval-Augmented Generation (RAG) system designed for educational content, specifically the CDER Parallel and Distributed Computing curriculum. It combines:

- **Neo4j Knowledge Graph**: Stores entities, relationships, and document structure
- **ChromaDB Vector Database**: Enables semantic similarity search
- **OpenAI GPT-3.5-turbo**: Powers answer generation and entity extraction
- **Four Retrieval Modes**: No-RAG, Vector-Only, Graph-Only, and Hybrid

### Key Features

- **Four Retrieval Strategies**: Compare different approaches side-by-side
- **Knowledge Graph**: Entity and relationship-based retrieval
- **Vector Search**: Semantic similarity-based retrieval
- **Hybrid Fusion**: Combines vector and graph results
- **Web Interface**: Chainlit-based interactive UI
- **Docker Support**: Easy deployment with Docker Compose

---

## Architecture

### System Components

```
┌─────────────────┐
│   User Query    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      CDER Chatbot (Orchestrator)    │
└────────┬────────────────────────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
    ▼         ▼          ▼          ▼
┌──────┐ ┌────────┐ ┌────────┐ ┌────────┐
│No-RAG│ │Vector  │ │ Graph  │ │ Hybrid │
│      │ │Retriever│ │Retriever│ │Retriever│
└──┬───┘ └───┬────┘ └───┬────┘ └───┬────┘
   │         │          │          │
   │         ▼          ▼          │
   │    ┌────────┐ ┌────────┐     │
   │    │ChromaDB│ │ Neo4j  │     │
   │    └────────┘ └────────┘     │
   │                               │
   └───────────────┬───────────────┘
                   ▼
            ┌─────────────┐
            │   OpenAI    │
            │ GPT-3.5-turbo│
            └─────────────┘
                   │
                   ▼
            ┌─────────────┐
            │   Answer    │
            └─────────────┘
```

### Data Flow

1. **Document Ingestion**: PDF/DOCX files → Document Processor
2. **Chunking**: Documents split into 512-token chunks with 50-token overlap
3. **Embedding**: Generate vectors using Sentence-Transformers (local) or OpenAI
4. **Vector Storage**: Store embeddings in ChromaDB with metadata
5. **Entity Extraction**: Extract entities and relationships using LLM
6. **Graph Construction**: Create nodes and edges in Neo4j
7. **Query Processing**: User query → Embedding + Entity extraction
8. **Retrieval**: Execute selected retrieval strategy
9. **Generation**: LLM synthesizes answer with retrieved context
10. **Response**: Display answer with citations and metadata

---

## Installation & Setup

### Prerequisites

- **Docker Desktop** (recommended) or Python 3.9+
- **Neo4j Aura** account (free tier available) or local Neo4j instance
- **OpenAI API Key** (for GPT-3.5-turbo)

### Quick Start with Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cder-graphrag
   ```

2. **Create `.env` file**
   ```bash
   # Neo4j Configuration
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
   NEO4J_DATABASE=neo4j

   # OpenAI Configuration
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-proj-your-api-key-here
   ```

3. **Build and start services**
   ```bash
   docker compose build
   docker compose up frontend -d
   ```

4. **Index documents** (first time only)
   ```bash
   docker compose run --rm cder-graphrag python index_documents.py
   ```

5. **Access the web interface**
   - Open http://localhost:8000 in your browser

### Manual Setup (Without Docker)

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Create `.env` file (same as Docker setup)

4. **Index documents**
   ```bash
   python index_documents.py
   ```

5. **Run the system**
   ```bash
   # Web interface
   chainlit run frontend/app.py

   # Or command line
   python main.py
   ```

---

## Configuration

### Environment Variables (`.env`)

| Variable | Description | Required |
|----------|-------------|----------|
| `NEO4J_URI` | Neo4j connection URI | Yes (for graph modes) |
| `NEO4J_USER` | Neo4j username | Yes (for graph modes) |
| `NEO4J_PASSWORD` | Neo4j password | Yes (for graph modes) |
| `NEO4J_DATABASE` | Database name | No (default: neo4j) |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `LLM_PROVIDER` | LLM provider | No (default: openai) |

### Configuration File (`config/config.yaml`)

#### LLM Settings

```yaml
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"  # Cheaper OpenAI model
  temperature: 0.3         # Low temperature for consistent answers
  max_tokens: 300          # Maximum response length
```

#### Retrieval Settings

```yaml
retrieval:
  vector_config:
    top_k: 3               # Number of vector results
  graph_config:
    max_hops: 2            # Graph traversal depth
  hybrid_config:
    vector_weight: 0.6     # Weight for vector results
    graph_weight: 0.4      # Weight for graph results
```

#### Processing Settings

```yaml
processing:
  chunk_size: 512         # Tokens per chunk
  chunk_overlap: 50       # Overlap between chunks
```

---

## Usage Guide

### Web Interface

1. **Start the frontend**
   ```bash
   docker compose up frontend
   ```

2. **Open browser**
   - Navigate to http://localhost:8000

3. **Ask questions**
   - Type your question in the chat interface
   - The system will show all 4 retrieval modes side-by-side
   - Compare answers, citations, and performance metrics

### Command Line Interface

#### Interactive Mode
```bash
python main.py
# or
python main.py --mode interactive
```

#### Single Query
```bash
python main.py --mode query --query "What is parallel computing?" --retrieval hybrid
```

#### Compare All Modes
```bash
python main.py --mode compare --query "Explain MapReduce"
```

#### Compare with Detailed Output
```bash
python compare_all_modes.py "What is parallel computing?"
```

### Available Commands (Interactive Mode)

- `/compare` - Compare all retrieval approaches
- `/stats` - Show system statistics
- `/history` - Show recent queries
- `/exit` - Quit

---

## API Reference

### CDERChatbot Class

Main orchestrator class for the RAG system.

#### Methods

##### `__init__(config_path='config/config.yaml', env_path='.env')`
Initialize the chatbot with configuration.

**Parameters:**
- `config_path` (str): Path to configuration YAML file
- `env_path` (str): Path to environment variables file

##### `process_query(user_query: str, retrieval_mode: str = "hybrid") -> Dict[str, Any]`
Process a single query through the specified retrieval mode.

**Parameters:**
- `user_query` (str): User's question
- `retrieval_mode` (str): One of `"no-rag"`, `"vector-only"`, `"graph-only"`, `"hybrid"`

**Returns:**
```python
{
    'query': str,
    'answer': str,
    'retrieval_mode': str,
    'metadata': {
        'retrieval': {
            'latency_ms': float,
            'chunks_retrieved': int,
            ...
        },
        'generation': {
            'tokens_used': dict,
            'latency_ms': float,
            'confidence': float
        },
        'citations': list
    }
}
```

##### `compare_all_retrievals(user_query: str) -> Dict[str, Any]`
Run query through all four retrieval approaches.

**Parameters:**
- `user_query` (str): User's question

**Returns:** Dictionary with results from all four modes

##### `interactive_chat()`
Start interactive command-line chat interface.

### Retrievers

#### NoRAGRetriever
Baseline retriever that returns empty context. Used for comparison.

#### VectorRetriever
Semantic similarity search using ChromaDB embeddings.

**Configuration:**
- `k`: Number of top results (default: 3)

#### GraphRetriever
Entity and relationship-based graph traversal using Neo4j.

**Configuration:**
- `hops`: Maximum graph traversal depth (default: 2)

#### HybridRetriever
Combines vector and graph retrieval with weighted fusion.

**Configuration:**
- `vector_weight`: Weight for vector results (default: 0.6)
- `graph_weight`: Weight for graph results (default: 0.4)
- `fusion_strategy`: `"ranked_union"` or `"weighted_sum"`

---

## Docker Deployment

### Docker Compose Services

#### `cder-graphrag`
Main application container.

**Usage:**
```bash
# Interactive shell
docker compose run --rm cder-graphrag

# Run specific script
docker compose run --rm cder-graphrag python main.py
```

#### `frontend`
Chainlit web interface.

**Usage:**
```bash
# Start frontend
docker compose up frontend

# View logs
docker compose logs frontend -f

# Stop frontend
docker compose stop frontend
```

### Docker Commands

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild without cache
docker compose build --no-cache
```

### Volume Mounts

- `.:/app` - Project code
- `./artifacts:/app/artifacts` - Persistent data (vector store, logs)

---

## Troubleshooting

### Common Issues

#### Neo4j Connection Failed

**Symptoms:**
- Error: "Neo4j connection failed"
- Graph-Only and Hybrid modes unavailable

**Solutions:**
1. Verify Neo4j Aura database is active
2. Check URI format: `neo4j+s://xxxxx.databases.neo4j.io`
3. Verify credentials in `.env` file
4. Check firewall/proxy settings (SSL connections)

#### Rate Limit Errors (429)

**Symptoms:**
- "429 Too Many Requests" errors
- Slow response times

**Solutions:**
1. System automatically retries with exponential backoff
2. Rate limiting: 2 seconds between requests
3. Each query makes exactly 4 API calls (one per mode)
4. Check OpenAI dashboard for usage limits

#### Empty Retrieval Results

**Symptoms:**
- No chunks retrieved
- Empty answers

**Solutions:**
1. Ensure documents are indexed: `python index_documents.py`
2. Check `data/cder_chapters/` contains PDF/DOCX files
3. Verify vector store exists: `artifacts/vector_store/`
4. Check logs for indexing errors

#### ChromaDB Errors

**Symptoms:**
- "Collection not found" errors
- Database corruption

**Solutions:**
1. Delete `artifacts/vector_store/` directory
2. Re-index documents: `python index_documents.py`
3. Check disk space

### Logs

View logs for debugging:

```bash
# Docker logs
docker compose logs frontend -f

# Application logs
tail -f artifacts/logs/app.log
```

---

## Performance Optimization

### API Call Optimization

- **Current**: 4 API calls per query (one per mode)
- **Rate Limiting**: 2 seconds between requests
- **Retry Logic**: Exponential backoff (2s → 4s → 8s → 16s → 30s)

### Token Limits

- **Input**: Max 8000 tokens (truncated to 6000 if exceeded)
- **Output**: Max 300 tokens per response
- **Context**: Automatically truncated if too large

### Retrieval Optimization

- **Vector**: Top 3 results (configurable)
- **Graph**: Top 10 chunks (reduced from 20)
- **Hybrid**: Combines both with weighted fusion

### Caching

- **Embeddings**: Cached in ChromaDB
- **Vector Store**: Persisted to disk
- **Graph**: Stored in Neo4j (persistent)

### Best Practices

1. **Index documents once** before first use
2. **Use Docker** for consistent environment
3. **Monitor API usage** in OpenAI dashboard
4. **Check logs** for performance issues
5. **Adjust `top_k`** based on your needs (lower = faster, less context)

---

## Additional Resources

- **Quick Start**: See `QUICK_START.md`
- **Docker Setup**: See `DOCKER_SETUP.md`
- **Frontend Guide**: See `FRONTEND_SETUP.md`
- **Usage Examples**: See `USAGE.md`

---

## License

This project is for educational purposes as part of the CDER curriculum.

---

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review logs in `artifacts/logs/`
3. Check Docker logs: `docker compose logs`

---

**Last Updated**: November 2025
**Version**: 1.0.0

