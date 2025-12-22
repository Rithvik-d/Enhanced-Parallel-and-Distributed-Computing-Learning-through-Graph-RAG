"""Check Neo4j graph structure to understand why graph retrieval isn't working."""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

database = os.getenv("NEO4J_DATABASE", "neo4j")

with driver.session(database=database) as session:
    print("=" * 60)
    print("Neo4j Graph Structure Analysis")
    print("=" * 60)
    
    # Check node types
    print("\n[1] Node Types:")
    result = session.run("""
        MATCH (n)
        RETURN DISTINCT labels(n) as labels, count(*) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"  {record['labels']}: {record['count']} nodes")
    
    # Check relationship types
    print("\n[2] Relationship Types:")
    result = session.run("""
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) as rel_type, count(*) as count
        ORDER BY count DESC
    """)
    for record in result:
        print(f"  {record['rel_type']}: {record['count']} relationships")
    
    # Check entity-chunk connections
    print("\n[3] Entity-Chunk Connections:")
    result = session.run("""
        MATCH (e:Entity)-[r]->(c:Chunk)
        RETURN type(r) as rel_type, count(*) as count
    """)
    found = False
    for record in result:
        print(f"  {record['rel_type']}: {record['count']} relationships")
        found = True
    if not found:
        print("  ❌ NO relationships found between Entity and Chunk nodes!")
    
    # Check sample entities
    print("\n[4] Sample Entities (first 10):")
    result = session.run("""
        MATCH (e:Entity)
        RETURN e.name, e.type
        LIMIT 10
    """)
    for record in result:
        print(f"  - {record['e.name']} ({record['e.type']})")
    
    # Check if entities have any relationships
    print("\n[5] Entity Relationships:")
    result = session.run("""
        MATCH (e:Entity)-[r]->(n)
        RETURN type(r) as rel_type, labels(n) as target_type, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    """)
    found = False
    for record in result:
        print(f"  {record['rel_type']} -> {record['target_type']}: {record['count']}")
        found = True
    if not found:
        print("  ❌ Entities have NO outgoing relationships!")
    
    # Check chunk relationships
    print("\n[6] Chunk Relationships:")
    result = session.run("""
        MATCH (c:Chunk)-[r]->(n)
        RETURN type(r) as rel_type, labels(n) as target_type, count(*) as count
        ORDER BY count DESC
        LIMIT 10
    """)
    for record in result:
        print(f"  {record['rel_type']} -> {record['target_type']}: {record['count']}")

driver.close()

