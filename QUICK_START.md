# Quick Start Guide

Get the CDER GraphRAG System running in 5 minutes!

## Prerequisites

- Docker Desktop installed and running
- Neo4j Aura account (free tier: https://neo4j.com/cloud/aura/)
- OpenAI API key (https://platform.openai.com/api-keys)

## Step 1: Setup Environment (2 minutes)

1. **Create `.env` file** in project root:

```env
# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# OpenAI Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-proj-your-api-key-here
```

2. **Get Neo4j Aura credentials**:
   - Sign up at https://neo4j.com/cloud/aura/
   - Create a free database
   - Copy connection URI, username, and password

3. **Get OpenAI API key**:
   - Sign up at https://platform.openai.com/
   - Go to API Keys section
   - Create new key

## Step 2: Build and Start (2 minutes)

```bash
# Build Docker image
docker compose build

# Start frontend
docker compose up frontend -d

# Check status
docker compose ps
```

Wait for: "Your app is available at http://0.0.0.0:8000"

## Step 3: Index Documents (1 minute)

```bash
# Add PDF files to data/cder_chapters/ (if not already there)

# Index documents
docker compose run --rm cder-graphrag python index_documents.py
```

**Note**: If you don't have documents yet, the system will still work but with limited context.

## Step 4: Use the System!

1. **Open browser**: http://localhost:8000
2. **Ask a question**: "What is parallel computing?"
3. **See results**: All 4 retrieval modes compared side-by-side!

## Common Commands

```bash
# View logs
docker compose logs frontend -f

# Stop
docker compose stop frontend

# Restart
docker compose restart frontend

# Run command line interface
docker compose run --rm cder-graphrag python main.py
```

## Troubleshooting

### "Neo4j connection failed"
- Check Neo4j Aura database is active
- Verify credentials in `.env`
- Ensure URI format: `neo4j+s://xxxxx.databases.neo4j.io`

### "Port 8000 already in use"
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead
```

### "No documents found"
- Add PDF files to `data/cder_chapters/`
- Run indexing: `docker compose run --rm cder-graphrag python index_documents.py`

## Next Steps

- Read [Complete Documentation](DOCUMENTATION.md)
- Check [API Reference](API_REFERENCE.md)
- See [Usage Examples](USAGE.md)

---

**That's it!** You're ready to use the CDER GraphRAG System. 🚀
