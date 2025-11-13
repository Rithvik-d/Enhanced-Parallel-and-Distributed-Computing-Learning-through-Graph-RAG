# Quick Start - Get All 4 Answers

## ✅ What's Ready

- ✅ All 4 retrieval modes working (No-RAG, Vector-Only, Graph-Only, Hybrid)
- ✅ Neo4j connection working in Docker
- ✅ Clean project structure (unnecessary files removed)

## 🚀 Get All 4 Answers

Run this command to see all 4 retrieval modes side by side:

```powershell
docker-compose run --rm cder-graphrag python compare_all_modes.py "What is parallel computing?"
```

Or using main.py:

```powershell
docker-compose run --rm cder-graphrag python main.py --mode all-modes --query "What is parallel computing?"
```

## 📊 What You'll Get

1. **NO-RAG (Baseline)** - Answer without retrieval
2. **VECTOR-ONLY RAG** - Semantic similarity search
3. **GRAPH-ONLY RAG** - Entity-based graph traversal
4. **HYBRID RAG** - Combined vector + graph retrieval

Each answer includes:
- Full answer text
- Citations
- Performance metrics (latency, tokens, confidence)
- Side-by-side comparison table

## 📁 Project Structure

Essential files only:
- `main.py` - Main entry point
- `compare_all_modes.py` - Compare all 4 modes
- `index_documents.py` - Index documents
- `src/` - Core modules
- `config/` - Configuration
- `data/` - Your PDF/DOCX files
- `Dockerfile` & `docker-compose.yml` - Docker setup

## 🎯 Other Commands

```powershell
# Interactive chatbot
docker-compose run --rm cder-graphrag python main.py

# Single query with specific mode
docker-compose run --rm cder-graphrag python main.py --mode query --query "question" --retrieval hybrid

# Index documents (first time)
docker-compose run --rm cder-graphrag python index_documents.py
```

