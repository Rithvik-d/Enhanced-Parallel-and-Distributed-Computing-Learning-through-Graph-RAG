# CDER GraphRAG System

A hybrid retrieval-augmented generation (RAG) system for educational content, specifically designed for the CDER Parallel and Distributed Computing curriculum. This system integrates Neo4j knowledge graphs with ChromaDB vector databases to provide four different retrieval approaches for question-answering.

## Features

- **Four Retrieval Strategies**:
  - **No-RAG**: Baseline without external retrieval
  - **Vector-Only**: Semantic similarity search using embeddings
  - **Graph-Only**: Entity and relationship-based graph traversal
  - **Hybrid**: Combined vector and graph retrieval with weighted fusion

- **Knowledge Graph**: Neo4j-based graph storing entities, relationships, and document structure
- **Vector Database**: ChromaDB for efficient similarity search
- **LLM Integration**: GPT-4 for answer generation and entity extraction
- **Interactive Chatbot**: Command-line interface for querying

## Project Structure

```
cder-graphrag/
├── config/
│   ├── config.yaml          # Main configuration file
│   └── .env.example          # Environment variables template
├── src/
│   ├── __init__.py
│   ├── config_loader.py      # Configuration management
│   ├── logger.py             # Logging setup
│   ├── doc_processor.py      # Document loading and chunking
│   ├── vector_db.py          # ChromaDB operations
│   ├── graph_db.py           # Neo4j operations
│   ├── entity_extractor.py   # LLM-based entity extraction
│   ├── llm_interface.py      # GPT-4 integration
│   ├── retrievers.py         # Four retrieval strategies
│   └── chatbot.py            # Main orchestrator
├── data/
│   └── cder_chapters/        # Place PDF/DOCX files here
├── notebooks/                # Jupyter notebooks for development
├── artifacts/                # Generated files (embeddings, logs)
├── tests/                    # Unit tests
├── main.py                   # Entry point
├── compare_all_modes.py      # Compare all 4 retrieval modes
├── index_documents.py        # Index documents into vector store
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
└── README.md                 # This file
```

## Installation

### Prerequisites

- Docker and Docker Compose (recommended for Neo4j support on Windows)
- OR Python 3.9+ with virtual environment
- Neo4j Aura account (cloud) - free tier available
- OpenAI API key

### Setup with Docker (Recommended)

1. **Clone or download the project**

2. **Configure environment variables**:
   - Create `.env` file in project root:
     ```
     NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=your-password
     NEO4J_DATABASE=neo4j
     OPENAI_API_KEY=sk-proj-your-key
     ```

3. **Build Docker image**:
   ```powershell
   docker-compose build
   ```

4. **Place documents**:
   - Add PDF or DOCX files to `data/cder_chapters/`

5. **Index documents**:
   ```powershell
   docker-compose run --rm cder-graphrag python index_documents.py
   ```

6. **Run the system**:
   ```powershell
   docker-compose run --rm cder-graphrag python main.py
   ```

### Setup without Docker (Windows - Vector-Only Mode)

1. **Create virtual environment**:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure environment variables** (same as Docker setup)

4. **Index documents**:
   ```powershell
   python index_documents.py
   ```

5. **Run the system**:
   ```powershell
   python main.py
   ```

**Note**: Neo4j connection may not work on Windows due to SSL/network issues. Use Docker for full functionality including graph and hybrid retrieval.

## Usage

### Get All 4 Answers (Recommended)

To see all 4 retrieval modes (No-RAG, Vector-Only, Graph-Only, Hybrid) side by side:

**Using Docker (Neo4j works!):**
```powershell
docker-compose run --rm cder-graphrag python compare_all_modes.py "What is parallel computing?"
```

**Or using main.py:**
```powershell
docker-compose run --rm cder-graphrag python main.py --mode all-modes --query "What is parallel computing?"
```

This will show:
- All 4 answers with full details
- Performance metrics (latency, tokens, confidence)
- Side-by-side comparison table

### Interactive Mode

Start the interactive chatbot:

```bash
python main.py
```

Or explicitly:

```bash
python main.py --mode interactive
```

Commands in interactive mode:
- `/compare` - Compare all retrieval approaches
- `/stats` - Show system statistics
- `/history` - Show recent queries
- `/exit` - Quit

### Single Query Mode

Process a single query:

```bash
python main.py --mode query --query "What is MapReduce?" --retrieval hybrid
```

### Comparison Mode

Compare all four retrieval approaches:

```bash
# Using Docker (recommended)
docker-compose run --rm cder-graphrag python main.py --mode compare --query "Explain parallel algorithms"

# Or get detailed comparison with all 4 answers
docker-compose run --rm cder-graphrag python compare_all_modes.py "Explain parallel algorithms"
```

## Configuration

Edit `config/config.yaml` to customize:

- **Document processing**: Chunk size, overlap
- **Embeddings**: Provider (OpenAI or Sentence-Transformers), model
- **Retrieval**: Top-k, graph hops, fusion weights
- **LLM**: Model, temperature, max tokens

## Development

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Jupyter Notebooks

Explore the system interactively:

```bash
jupyter notebook notebooks/
```

## Architecture

### Data Flow

1. **Document Ingestion**: Load PDF/DOCX files
2. **Chunking**: Split into 512-token chunks with 50-token overlap
3. **Embedding**: Generate vectors (OpenAI or Sentence-Transformers)
4. **Vector Storage**: Store in ChromaDB with metadata
5. **Entity Extraction**: Use LLM to extract entities and relationships
6. **Graph Construction**: Create nodes and edges in Neo4j
7. **Query Processing**: Convert query to embedding and extract entities
8. **Retrieval**: Execute selected retrieval strategy
9. **Generation**: LLM synthesizes answer with retrieved context
10. **Response**: Display answer with citations and metadata

### Retrieval Strategies

#### Vector-Only
- Embeds query and searches ChromaDB for similar chunks
- Returns top-k results ranked by cosine similarity

#### Graph-Only
- Extracts entities from query
- Traverses Neo4j graph from seed entities
- Collects connected chunks via relationships

#### Hybrid
- Runs vector and graph retrieval in parallel
- Fuses results using weighted scoring
- Deduplicates and ranks final context

## Performance

Expected metrics (approximate):
- **Vector RAG Accuracy**: 60-75%
- **Graph RAG Accuracy**: 70-80%
- **Hybrid RAG Accuracy**: 85-90%
- **Retrieval Latency**: <2 seconds (hybrid)
- **Hallucination Reduction**: 80%+ vs. No-RAG baseline

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Neo4j connection fails | Verify URI, username, password in `.env` |
| Slow embeddings | Use GPU or OpenAI API (faster than local) |
| Empty retrieval results | Check if documents are loaded and indexed |
| High API costs | Use Sentence-Transformers for local embeddings |
| Memory errors | Reduce chunk size or batch size in config |

### Logs

Check logs in `artifacts/logs/app.log` for detailed error messages.

## License

This project is for educational purposes as part of the CDER curriculum.

## Contributing

This is an academic project. For improvements or bug fixes, please follow standard Python best practices and include tests.

## Acknowledgments

- CDER Parallel and Distributed Computing curriculum
- Neo4j for graph database
- ChromaDB for vector storage
- OpenAI for LLM capabilities
- LangChain for document processing

