"""Test graph retrieval to debug why chunks aren't being retrieved."""

import os
from dotenv import load_dotenv
from src.graph_db import GraphDBManager
from src.llm_interface import LLMInterface

load_dotenv()

# Initialize components
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
neo4j_password = os.getenv("NEO4J_PASSWORD")
neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

graph_manager = GraphDBManager(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)

# Check OpenAI key
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    print("WARNING: OPENAI_API_KEY not found. Entity extraction will be skipped.")
    llm_interface = None
else:
    llm_interface = LLMInterface(api_key=openai_key)

query = "What is parallel computing?"

print("=" * 60)
print("Graph Retrieval Debug Test")
print("=" * 60)
print(f"Query: {query}\n")

# Step 1: Extract entities from query
print("[STEP 1] Extracting entities from query...")
if llm_interface:
    entity_names = llm_interface.extract_entities_from_query(query)
    print(f"Extracted entities: {entity_names}\n")
else:
    entity_names = []
    print("Skipping entity extraction (no OpenAI key)\n")

# Step 2: Search for entities in graph
print("[STEP 2] Searching for entities in graph...")
entity_ids = graph_manager.entity_seed_search(query)
print(f"Found entity IDs from query search: {entity_ids}")

for entity_name in entity_names:
    additional_ids = graph_manager.entity_seed_search(entity_name)
    print(f"Found entity IDs for '{entity_name}': {additional_ids}")
    entity_ids.extend(additional_ids)

# Remove duplicates
entity_ids = list(dict.fromkeys(entity_ids))
print(f"\nTotal unique entity IDs: {len(entity_ids)}")
if entity_ids:
    print(f"Entity IDs: {entity_ids[:10]}")  # Show first 10
else:
    print("❌ No entities found! This is why no chunks are retrieved.\n")
    print("Checking what entities exist in the graph...")
    with graph_manager.driver.session(database=graph_manager.database) as session:
        result = session.run("MATCH (e:Entity) RETURN e.name, e.type LIMIT 20")
        print("\nSample entities in graph:")
        for record in result:
            print(f"  - {record['e.name']} ({record['e.type']})")
    
    # Check if chunks are connected to entities
    print("\nChecking entity-chunk relationships...")
    with graph_manager.driver.session(database=graph_manager.database) as session:
        result = session.run("""
            MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
            RETURN count(*) as count
        """)
        count = result.single()['count']
        print(f"Total entity-chunk relationships: {count}")
    
    exit(1)

# Step 3: Graph expansion
print(f"\n[STEP 3] Performing graph expansion from {len(entity_ids)} entities...")
expansion_result = graph_manager.graph_expansion(entity_ids, hops=2)
chunk_ids = expansion_result['chunk_ids']

print(f"Found chunk IDs: {len(chunk_ids)}")
if chunk_ids:
    print(f"First 10 chunk IDs: {chunk_ids[:10]}")
else:
    print("❌ No chunks found after graph expansion!")
    print("\nChecking entity connections...")
    with graph_manager.driver.session(database=graph_manager.database) as session:
        # Check if entities have any relationships
        for eid in entity_ids[:5]:  # Check first 5
            result = session.run("""
                MATCH (e:Entity {id: $eid})-[r]->(n)
                RETURN type(r) as rel_type, labels(n) as node_labels, count(*) as count
            """, eid=eid)
            print(f"\nEntity {eid} relationships:")
            for record in result:
                print(f"  - {record['rel_type']} -> {record['node_labels']}: {record['count']}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)

graph_manager.close()

