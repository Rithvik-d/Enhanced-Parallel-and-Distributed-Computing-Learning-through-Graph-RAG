"""
Index documents into the vector database.
Run this to populate ChromaDB with your document embeddings.
"""

from src.chatbot import CDERChatbot
from src.doc_processor import DocumentProcessor
from src.vector_db import VectorDBManager
from src.config_loader import ConfigLoader
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("Document Indexing - CDER GraphRAG")
print("=" * 60)
print()

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load()
runtime_config = config_loader.get_runtime_config()

# Initialize components
print("1. Initializing components...")
doc_processor = DocumentProcessor(
    chunk_size=config.processing.chunk_size,
    chunk_overlap=config.processing.chunk_overlap
)

# Only pass API key if using OpenAI embeddings
embedding_api_key = None
if config.embeddings.provider == "openai":
    embedding_api_key = runtime_config.get('openai_api_key')

vector_manager = VectorDBManager(
    collection_name="cder_embeddings",
    persist_dir="./artifacts/vector_store",
    embedding_provider=config.embeddings.provider,
    embedding_model=config.embeddings.model,
    api_key=embedding_api_key
)

print("   [OK] Components initialized\n")

# Load documents
print("2. Loading documents...")
doc_dir = config.documents.source_dir
documents = doc_processor.load_all_documents(doc_dir)
print(f"   [OK] Loaded {len(documents)} documents\n")

# Process and index each document
print("3. Processing and indexing documents...")
total_chunks = 0

for i, doc in enumerate(documents, 1):
    print(f"\n   Processing document {i}/{len(documents)}: {doc.metadata.get('source', 'Unknown')}")
    
    # Chunk document
    chunks = doc_processor.chunk_document(doc)
    print(f"   Created {len(chunks)} chunks")
    
    # Add to vector database
    chunk_ids = vector_manager.add_documents(chunks, batch_size=5)
    total_chunks += len(chunk_ids)
    print(f"   [OK] Indexed {len(chunk_ids)} chunks")

print(f"\n[SUCCESS] Indexing complete!")
print(f"   Total documents: {len(documents)}")
print(f"   Total chunks indexed: {total_chunks}")

# Get collection stats
stats = vector_manager.get_collection_stats()
print(f"\nVector Database Stats:")
print(f"   Total documents in collection: {stats.get('total_documents', 0)}")
print(f"   Embedding dimensions: {stats.get('embedding_dimensions', 'N/A')}")

print("\n" + "=" * 60)
print("Documents are now indexed and ready for queries!")
print("=" * 60)
print("\nTry running:")
print("  python test_query.py")
print("  python main.py")

