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
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- Neo4j Aura account (cloud) or local Neo4j instance
- OpenAI API key

### Setup

1. **Clone or download the project**

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env` (or create manually)
   - Edit `.env` with your credentials:
     ```
     NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=your-password
     OPENAI_API_KEY=sk-proj-your-key
     ```

5. **Configure Neo4j**:
   - **Option 1 (Cloud)**: Create a free Neo4j Aura account at https://neo4j.com/cloud/aura/
   - **Option 2 (Local)**: Run Neo4j via Docker:
     ```bash
     docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:latest
     ```

6. **Place documents**:
   - Add PDF or DOCX files to `data/cder_chapters/`

## Usage

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
python main.py --mode compare --query "Explain parallel algorithms"
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

