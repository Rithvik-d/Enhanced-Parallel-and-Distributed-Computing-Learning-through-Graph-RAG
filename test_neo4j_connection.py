"""Test Neo4j connection from Docker."""
import socket
import os
from dotenv import load_dotenv

load_dotenv()

neo4j_uri = os.getenv('NEO4J_URI', '')
print(f"Testing Neo4j URI: {neo4j_uri}")

# Extract hostname from URI
if '://' in neo4j_uri:
    hostname = neo4j_uri.split('://')[1].split(':')[0].split('/')[0]
else:
    hostname = neo4j_uri

print(f"Hostname: {hostname}")

# Test DNS resolution
try:
    print("\n1. Testing DNS resolution...")
    addr = socket.gethostbyname(hostname)
    print(f"   ✓ DNS resolved to: {addr}")
except Exception as e:
    print(f"   ✗ DNS resolution failed: {e}")
    print("   This is likely a Docker network/DNS issue")
    exit(1)

# Test Neo4j connection
try:
    print("\n2. Testing Neo4j connection...")
    from src.graph_db import GraphDBManager
    
    gm = GraphDBManager(
        uri=os.getenv('NEO4J_URI'),
        user=os.getenv('NEO4J_USER'),
        password=os.getenv('NEO4J_PASSWORD'),
        database=os.getenv('NEO4J_DATABASE', 'neo4j')
    )
    print("   ✓ Neo4j connection successful!")
    
    # Test a simple query
    with gm.driver.session() as session:
        result = session.run("RETURN 1 as test")
        record = result.single()
        print(f"   ✓ Query test successful: {record['test']}")
    
except Exception as e:
    print(f"   ✗ Neo4j connection failed: {e}")
    exit(1)
