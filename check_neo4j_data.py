"""Check what data exists in Neo4j graph database."""
from src.graph_db import GraphDBManager
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 60)
print("Neo4j Graph Database Status Check")
print("=" * 60)
print()

# Connect to Neo4j
try:
    gm = GraphDBManager(
        uri=os.getenv('NEO4J_URI'),
        user=os.getenv('NEO4J_USER'),
        password=os.getenv('NEO4J_PASSWORD'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j')
    )
    print("✓ Connected to Neo4j\n")
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    exit(1)

# Check node counts
with gm.driver.session() as session:
    # Total nodes
    result = session.run('MATCH (n) RETURN count(n) as count')
    total_nodes = result.single()['count']
    print(f"Total nodes: {total_nodes}")
    
    # Document nodes
    result = session.run('MATCH (d:Document) RETURN count(d) as count')
    doc_nodes = result.single()['count']
    print(f"Document nodes: {doc_nodes}")
    
    # Chunk nodes
    result = session.run('MATCH (c:Chunk) RETURN count(c) as count')
    chunk_nodes = result.single()['count']
    print(f"Chunk nodes: {chunk_nodes}")
    
    # Entity nodes
    result = session.run('MATCH (e:Entity) RETURN count(e) as count')
    entity_nodes = result.single()['count']
    print(f"Entity nodes: {entity_nodes}")
    
    # Relationships
    result = session.run('MATCH ()-[r]->() RETURN count(r) as count')
    relationships = result.single()['count']
    print(f"Total relationships: {relationships}")
    
    # HAS_ENTITY relationships (chunk -> entity)
    result = session.run('MATCH ()-[r:HAS_ENTITY]->() RETURN count(r) as count')
    has_entity = result.single()['count']
    print(f"HAS_ENTITY relationships: {has_entity}")
    
    # PART_OF relationships (chunk -> document)
    result = session.run('MATCH ()-[r:PART_OF]->() RETURN count(r) as count')
    part_of = result.single()['count']
    print(f"PART_OF relationships: {part_of}")
    
    print()
    
    # Sample entities
    if entity_nodes > 0:
        print("Sample entities:")
        result = session.run('MATCH (e:Entity) RETURN e.name as name, e.type as type LIMIT 10')
        for record in result:
            print(f"  - {record['name']} ({record['type']})")
    
    print()
    
    # Test graph retrieval
    print("Testing graph retrieval...")
    query = "parallel computing"
    entity_ids = gm.entity_seed_search(query)
    print(f"Found {len(entity_ids)} entities matching '{query}'")
    
    if entity_ids:
        print(f"Entity IDs: {entity_ids[:5]}...")  # Show first 5
        
        # Test graph expansion
        expansion = gm.graph_expansion(entity_ids[:3], hops=2)
        print(f"Graph expansion found:")
        print(f"  - {len(expansion['chunk_ids'])} chunks")
        print(f"  - {len(expansion['entities'])} entities")
        print(f"  - {len(expansion['relationships'])} relationships")
    else:
        print("⚠️ No entities found - graph retrieval will not work")
        print("   You need to run index_graph.py to populate the graph")

print("\n" + "=" * 60)

