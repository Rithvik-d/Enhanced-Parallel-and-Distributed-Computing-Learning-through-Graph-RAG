"""
Index documents into Neo4j graph database.
Extracts entities, creates nodes and relationships for graph retrieval.
"""

from src.chatbot import CDERChatbot
from src.doc_processor import DocumentProcessor
from src.graph_db import GraphDBManager
from src.entity_extractor import EntityExtractor
from src.config_loader import ConfigLoader
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

print("=" * 60)
print("Graph Database Indexing - CDER GraphRAG")
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

# Initialize Neo4j connection
try:
    graph_manager = GraphDBManager(
        uri=runtime_config['neo4j_uri'],
        user=runtime_config['neo4j_user'],
        password=runtime_config['neo4j_password'],
        database=runtime_config['neo4j_database']
    )
    print("   [OK] Neo4j connected")
except Exception as e:
    print(f"   [ERROR] Failed to connect to Neo4j: {e}")
    exit(1)

# Initialize entity extractor
entity_extractor = EntityExtractor(
    llm_model=config.llm.model,
    api_key=runtime_config['openai_api_key']
)
print("   [OK] Components initialized\n")

# Load documents
print("2. Loading documents...")
doc_dir = config.documents.source_dir
documents = doc_processor.load_all_documents(doc_dir)
print(f"   [OK] Loaded {len(documents)} documents\n")

# Process and index each document into graph
print("3. Processing and indexing documents into graph...")
total_entities = 0
total_chunks = 0

for i, doc in enumerate(documents, 1):
    print(f"\n   Processing document {i}/{len(documents)}: {doc.metadata.get('source', 'Unknown')}")
    
    # Extract document ID from filename
    doc_source = doc.metadata.get('source', '')
    doc_id = Path(doc_source).stem if doc_source else f"doc_{i}"
    
    # Create document node
    doc_metadata = {
        'title': doc.metadata.get('title', doc_id),
        'filename': Path(doc_source).name if doc_source else f"{doc_id}.pdf",
        'chapter_number': doc.metadata.get('chapter', i),
        'word_count': len(doc.page_content.split())
    }
    graph_manager.add_document_node(doc_id, doc_metadata)
    print(f"   Created document node: {doc_id}")
    
    # Chunk document
    chunks = doc_processor.chunk_document(doc)
    print(f"   Created {len(chunks)} chunks")
    
    # Add chunk nodes to graph
    chunk_ids = graph_manager.add_chunk_nodes(chunks, doc_id)
    total_chunks += len(chunk_ids)
    print(f"   Created {len(chunk_ids)} chunk nodes")
    
    # Extract entities from chunks and link to graph
    print(f"   Extracting entities from chunks...")
    entities_created = 0
    
    for chunk_idx, chunk in enumerate(chunks):
        if chunk_idx % 10 == 0:
            print(f"      Processing chunk {chunk_idx + 1}/{len(chunks)}...")
        
        # Extract entities from chunk
        try:
            entities = entity_extractor.extract_entities(chunk.page_content)
            
            chunk_id = chunk.metadata.get('chunk_id', chunk_ids[chunk_idx] if chunk_idx < len(chunk_ids) else f"chunk_{chunk_idx}")
            
            for entity in entities:
                entity_name = entity['name']
                entity_type = entity.get('type', 'CONCEPT')
                entity_desc = entity.get('description', '')
                
                # Create or update entity node
                entity_node_id = graph_manager.add_entity_node(
                    name=entity_name,
                    entity_type=entity_type,
                    description=entity_desc
                )
                
                # Link chunk to entity
                graph_manager.link_chunk_to_entity(chunk_id, entity_node_id)
                entities_created += 1
                
        except Exception as e:
            print(f"      Warning: Failed to extract entities from chunk {chunk_idx}: {e}")
            continue
    
    total_entities += entities_created
    print(f"   [OK] Created {entities_created} entity relationships")

print(f"\n[SUCCESS] Graph indexing complete!")
print(f"   Total documents: {len(documents)}")
print(f"   Total chunks indexed: {total_chunks}")
print(f"   Total entity relationships: {total_entities}")

# Get graph stats
stats = graph_manager.get_graph_stats()
print(f"\nGraph Database Stats:")
print(f"   Total nodes: {stats.get('total_nodes', 0)}")
print(f"   Document nodes: {stats.get('document_nodes', 0)}")
print(f"   Chunk nodes: {stats.get('chunk_nodes', 0)}")
print(f"   Entity nodes: {stats.get('entity_nodes', 0)}")
print(f"   Total relationships: {stats.get('total_relationships', 0)}")

print("\n" + "=" * 60)
print("Graph database is now indexed and ready for graph retrieval!")
print("=" * 60)

