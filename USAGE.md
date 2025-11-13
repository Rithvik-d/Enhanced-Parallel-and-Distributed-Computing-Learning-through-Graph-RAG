# CDER GraphRAG - Usage Guide

## Quick Start

### Using Docker (Recommended - Neo4j Works!)

```powershell
# Run interactive chatbot
docker-compose run --rm cder-graphrag python main.py

# Get all 4 answers for a query
docker-compose run --rm cder-graphrag python main.py --mode all-modes --query "What is parallel computing?"

# Or use the comparison script directly
docker-compose run --rm cder-graphrag python compare_all_modes.py "What is parallel computing?"
```

### Using Windows (Vector-Only Mode)

```powershell
# Run interactive chatbot
python main.py

# Single query
python main.py --mode query --query "your question" --retrieval vector-only
```

## All 4 Retrieval Modes

The system provides 4 retrieval approaches:

1. **NO-RAG**: Baseline without external retrieval
2. **VECTOR-ONLY**: Semantic similarity search using embeddings
3. **GRAPH-ONLY**: Entity and relationship-based graph traversal
4. **HYBRID**: Combined vector and graph retrieval with weighted fusion

## Get All 4 Answers

To see all 4 answers side by side:

```powershell
# Using Docker (recommended)
docker-compose run --rm cder-graphrag python compare_all_modes.py "your question"

# Or using main.py
docker-compose run --rm cder-graphrag python main.py --mode all-modes --query "your question"
```

## Commands Reference

### Docker Commands

```powershell
# Build image (first time only)
docker-compose build

# Run interactive chatbot
docker-compose run --rm cder-graphrag python main.py

# Compare all 4 modes
docker-compose run --rm cder-graphrag python compare_all_modes.py "your question"

# Single query with specific mode
docker-compose run --rm cder-graphrag python main.py --mode query --query "question" --retrieval hybrid
```

### Windows Commands (Vector-Only)

```powershell
# Interactive chatbot
python main.py

# Single query
python main.py --mode query --query "question" --retrieval vector-only
```

## Configuration

Edit `config/config.yaml` and `.env` to customize:
- Chunk size and overlap
- Embedding model
- Retrieval parameters
- LLM settings

## Project Structure

```
cder-graphrag/
├── src/              # Core modules
├── config/           # Configuration files
├── data/             # Document files
├── artifacts/        # Generated files
├── main.py          # Entry point
├── compare_all_modes.py  # Compare all 4 modes
├── Dockerfile       # Docker configuration
└── docker-compose.yml
```

