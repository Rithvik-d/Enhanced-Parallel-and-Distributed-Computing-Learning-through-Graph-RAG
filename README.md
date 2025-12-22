# CDER GraphRAG System

A hybrid retrieval-augmented generation (RAG) system for educational content, specifically designed for the CDER Parallel and Distributed Computing curriculum. This system integrates Neo4j knowledge graphs with ChromaDB vector databases to provide four different retrieval approaches for question-answering.

## 🚀 Quick Start

### With Docker (Recommended)

```bash
# 1. Create .env file with your API keys
cp .env.example .env
# Edit .env with your Neo4j and OpenAI credentials

# 2. Build and start
docker compose build
docker compose up frontend -d

# 3. Index documents (first time only)
docker compose run --rm cder-graphrag python index_documents.py

# 4. Open http://localhost:8000
```

### Without Docker

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure .env file
# 3. Index documents
python index_documents.py

# 4. Run
chainlit run frontend/app.py
```

## ✨ Features

- **Four Retrieval Strategies**:
  - **No-RAG**: Baseline without external retrieval
  - **Vector-Only**: Semantic similarity search using embeddings
  - **Graph-Only**: Entity and relationship-based graph traversal
  - **Hybrid**: Combined vector and graph retrieval with weighted fusion

- **Knowledge Graph**: Neo4j-based graph storing entities, relationships, and document structure
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM Integration**: OpenAI GPT-3.5-turbo for answer generation
- **Web Interface**: Chainlit-based interactive UI showing all 4 modes side-by-side
- **Docker Support**: Easy deployment with Docker Compose

## 📋 Requirements

- Docker Desktop (recommended) or Python 3.9+
- Neo4j Aura account (free tier available)
- OpenAI API key

## 📁 Project Structure

```
cder-graphrag/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── chatbot.py            # Main orchestrator
│   ├── retrievers.py         # Four retrieval strategies
│   ├── vector_db.py          # ChromaDB operations
│   ├── graph_db.py           # Neo4j operations
│   ├── llm_interface.py      # OpenAI integration
│   └── ...
├── data/
│   └── cder_chapters/        # PDF/DOCX files here
├── frontend/
│   └── app.py               # Chainlit web interface
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 🔧 Configuration

### Environment Variables (`.env`)

```env
# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-key-here
LLM_PROVIDER=openai
```

### Config File (`config/config.yaml`)

Key settings:
- `llm.model`: `gpt-3.5-turbo` (cheaper model)
- `llm.temperature`: `0.3` (low for consistent answers)
- `retrieval.vector_config.top_k`: `3` (number of results)
- `retrieval.graph_config.max_hops`: `2` (graph depth)

## 📖 Usage

### Web Interface

```bash
docker compose up frontend
# Open http://localhost:8000
```

Ask questions and see all 4 retrieval modes compared side-by-side!

### Command Line

```bash
# Interactive mode
python main.py

# Single query
python main.py --mode query --query "What is parallel computing?" --retrieval hybrid

# Compare all modes
python main.py --mode compare --query "Explain MapReduce"
```

## 🐳 Docker Commands

```bash
# Build
docker compose build

# Start frontend
docker compose up frontend -d

# View logs
docker compose logs frontend -f

# Run scripts
docker compose run --rm cder-graphrag python main.py

# Stop
docker compose down
```

## 📊 Performance

- **API Calls**: 4 per query (one per mode)
- **Rate Limiting**: 2 seconds between requests
- **Response Time**: ~2-10 seconds per query (depending on mode)
- **Token Limits**: 8000 input, 300 output

## 🛠️ Troubleshooting

### Neo4j Connection Failed
- Verify Neo4j Aura database is active
- Check credentials in `.env`
- Ensure URI format: `neo4j+s://xxxxx.databases.neo4j.io`

### Rate Limit Errors
- System automatically retries with exponential backoff
- Check OpenAI dashboard for usage limits

### Empty Results
- Ensure documents are indexed: `python index_documents.py`
- Check `data/cder_chapters/` contains PDF files

## 📚 Documentation

- **[Complete Documentation](DOCUMENTATION.md)** - Full system documentation
- **[Quick Start Guide](QUICK_START.md)** - Get started quickly
- **[Docker Setup](DOCKER_SETUP.md)** - Docker deployment guide
- **[Usage Guide](USAGE.md)** - Detailed usage examples

## 🏗️ Architecture

```
User Query
    ↓
CDER Chatbot (Orchestrator)
    ↓
┌─────────┬──────────┬──────────┬──────────┐
│ No-RAG  │ Vector   │  Graph   │  Hybrid  │
└─────────┴─────┬────┴─────┬────┴──────────┘
                │          │
            ChromaDB    Neo4j
                │          │
                └─────┬────┘
                      ↓
                  OpenAI
                  GPT-3.5-turbo
                      ↓
                    Answer
```

## 📝 License

This project is for educational purposes as part of the CDER curriculum.

## 🤝 Contributing

This is an academic project. For improvements or bug fixes, please follow standard Python best practices and include tests.

---

**Version**: 1.0.0  
**Last Updated**: November 2025
