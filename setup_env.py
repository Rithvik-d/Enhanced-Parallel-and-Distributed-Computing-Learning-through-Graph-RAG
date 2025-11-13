"""
Helper script to create .env file from template.
Run this script to set up your environment variables.
"""

from pathlib import Path

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_example = Path("config/.env.example")
    env_file = Path(".env")
    
    if env_file.exists():
        print(".env file already exists. Skipping creation.")
        return
    
    if not env_example.exists():
        print("Error: config/.env.example not found. Creating basic template...")
        env_content = """# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-generated-password-here
NEO4J_DATABASE=neo4j

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-your-api-key-here

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
BATCH_SIZE=10

# Retrieval Parameters
VECTOR_SEARCH_K=5
GRAPH_SEARCH_HOPS=2
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_GRAPH_WEIGHT=0.4

# LLM Configuration
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3
MAX_RESPONSE_TOKENS=300

# Storage Paths
ARTIFACT_DIR=./artifacts
VECTOR_DB_PATH=./artifacts/vector_store
PERSIST_DIRECTORY=./artifacts/persist
DATA_DIR=./data/cder_chapters

# Logging
LOG_LEVEL=INFO
LOG_FILE=./artifacts/logs/app.log
"""
        env_file.write_text(env_content)
    else:
        # Copy from example
        env_content = env_example.read_text()
        env_file.write_text(env_content)
    
    print(f".env file created at {env_file.absolute()}")
    print("Please edit .env with your actual credentials:")
    print("  - Neo4j URI, username, and password")
    print("  - OpenAI API key")

if __name__ == "__main__":
    create_env_file()

