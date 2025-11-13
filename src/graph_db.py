"""
Neo4j graph database management for CDER GraphRAG system.
Handles knowledge graph creation, entity storage, and graph queries.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
import logging
from datetime import datetime

from neo4j import GraphDatabase, Driver, Session
from langchain.schema import Document as LangChainDocument

from src.logger import setup_logger

logger = setup_logger(__name__)


class GraphDBManager:
    """
    Manage Neo4j operations for knowledge graph.
    Handles nodes, relationships, and graph traversal queries.
    """
    
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            database: Database name (default: "neo4j")
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        
        # Connect to Neo4j
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connection
            self.verify_connection()
            logger.info(f"Connected to Neo4j: {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def verify_connection(self) -> bool:
        """
        Verify Neo4j connection is working.
        
        Returns:
            True if connection is valid
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Neo4j connection verified")
            return True
        except Exception as e:
            logger.error(f"Neo4j connection verification failed: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j: {e}")
    
    def create_schema(self) -> None:
        """
        Create graph schema: indexes and constraints.
        """
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique IDs
            constraints = [
                "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            ]
            
            # Create indexes for faster lookups
            indexes = [
                "CREATE INDEX document_title IF NOT EXISTS FOR (d:Document) ON (d.title)",
                "CREATE INDEX chunk_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_index)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint may already exist: {e}")
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index may already exist: {e}")
            
            logger.info("Graph schema created successfully")
    
    def add_document_node(
        self,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Create document node in graph.
        
        Args:
            doc_id: Unique document identifier
            metadata: Document metadata dictionary
            
        Returns:
            Document node ID
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MERGE (d:Document {id: $doc_id})
            SET d.title = $title,
                d.chapter_number = $chapter_number,
                d.filename = $filename,
                d.created_at = $created_at,
                d.word_count = $word_count
            RETURN d.id as id
            """
            
            result = session.run(
                query,
                doc_id=doc_id,
                title=metadata.get('title', 'Unknown'),
                chapter_number=metadata.get('chapter_number'),
                filename=metadata.get('filename', ''),
                created_at=datetime.now().isoformat(),
                word_count=metadata.get('word_count', 0)
            )
            
            record = result.single()
            node_id = record['id'] if record else doc_id
            logger.info(f"Created document node: {doc_id}")
            return node_id
    
    def add_chunk_nodes(
        self,
        chunks: List[LangChainDocument],
        parent_doc_id: str
    ) -> List[str]:
        """
        Create chunk nodes linked to parent document.
        
        Args:
            chunks: List of chunked documents
            parent_doc_id: Parent document ID
            
        Returns:
            List of chunk IDs created
        """
        chunk_ids = []
        
        with self.driver.session(database=self.database) as session:
            for chunk in chunks:
                chunk_id = chunk.metadata.get('chunk_id', f"chunk_{len(chunk_ids)}")
                
                query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text,
                    c.chunk_index = $chunk_index,
                    c.token_count = $token_count,
                    c.created_at = $created_at
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (c)-[r:PART_OF]->(d)
                SET r.order = $chunk_index,
                    r.position_in_doc = $chunk_index
                RETURN c.id as id
                """
                
                result = session.run(
                    query,
                    chunk_id=chunk_id,
                    text=chunk.page_content,
                    chunk_index=chunk.metadata.get('chunk_index', 0),
                    token_count=chunk.metadata.get('token_count', 0),
                    created_at=datetime.now().isoformat(),
                    doc_id=parent_doc_id
                )
                
                record = result.single()
                if record:
                    chunk_ids.append(record['id'])
            
            # Create sequential relationships between chunks
            self._add_chunk_relationships(chunk_ids, session)
        
        logger.info(f"Created {len(chunk_ids)} chunk nodes for document {parent_doc_id}")
        return chunk_ids
    
    def _add_chunk_relationships(
        self,
        chunk_ids: List[str],
        session: Session
    ) -> None:
        """
        Create NEXT_CHUNK relationships between sequential chunks.
        
        Args:
            chunk_ids: List of chunk IDs in order
            session: Neo4j session
        """
        for i in range(len(chunk_ids) - 1):
            query = """
            MATCH (c1:Chunk {id: $chunk_id1})
            MATCH (c2:Chunk {id: $chunk_id2})
            MERGE (c1)-[r:NEXT_CHUNK]->(c2)
            SET r.distance = 1
            """
            
            session.run(
                query,
                chunk_id1=chunk_ids[i],
                chunk_id2=chunk_ids[i + 1]
            )
    
    def add_entity_node(
        self,
        name: str,
        entity_type: str,
        description: Optional[str] = None
    ) -> str:
        """
        Create or update entity node.
        
        Args:
            name: Entity name
            entity_type: Entity type (ALGORITHM, TECHNOLOGY, CONCEPT, etc.)
            description: Optional entity description
            
        Returns:
            Entity node ID
        """
        entity_id = f"entity_{name.lower().replace(' ', '_')}"
        
        with self.driver.session(database=self.database) as session:
            query = """
            MERGE (e:Entity {id: $entity_id})
            ON CREATE SET e.name = $name,
                          e.type = $type,
                          e.description = $description,
                          e.frequency = 1,
                          e.created_at = $created_at
            ON MATCH SET e.frequency = e.frequency + 1,
                         e.description = COALESCE($description, e.description)
            RETURN e.id as id
            """
            
            result = session.run(
                query,
                entity_id=entity_id,
                name=name,
                type=entity_type,
                description=description,
                created_at=datetime.now().isoformat()
            )
            
            record = result.single()
            return record['id'] if record else entity_id
    
    def link_chunk_to_entity(
        self,
        chunk_id: str,
        entity_id: str,
        frequency: int = 1,
        context: Optional[str] = None
    ) -> None:
        """
        Create relationship between chunk and entity.
        
        Args:
            chunk_id: Chunk identifier
            entity_id: Entity identifier
            frequency: Frequency of entity in chunk
            context: Optional context snippet
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (e:Entity {id: $entity_id})
            MERGE (c)-[r:HAS_ENTITY]->(e)
            SET r.frequency = COALESCE(r.frequency, 0) + $frequency,
                r.context = COALESCE($context, r.context)
            """
            
            session.run(
                query,
                chunk_id=chunk_id,
                entity_id=entity_id,
                frequency=frequency,
                context=context
            )
    
    def add_entity_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        strength: float = 1.0
    ) -> None:
        """
        Create relationship between two entities.
        
        Args:
            source_entity_id: Source entity ID
            target_entity_id: Target entity ID
            relationship_type: Type of relationship (USES, ENABLES, etc.)
            strength: Relationship strength score
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (e1:Entity {id: $source_id})
            MATCH (e2:Entity {id: $target_id})
            MERGE (e1)-[r:RELATES_TO]->(e2)
            SET r.relationship_type = $rel_type,
                r.strength = $strength
            """
            
            session.run(
                query,
                source_id=source_entity_id,
                target_id=target_entity_id,
                rel_type=relationship_type,
                strength=strength
            )
    
    def entity_seed_search(self, query: str) -> List[str]:
        """
        Find entity nodes matching query terms.
        
        Args:
            query: Query text to search for
            
        Returns:
            List of entity IDs matching query
        """
        query_terms = query.lower().split()
        entity_ids = []
        
        with self.driver.session(database=self.database) as session:
            # Search for entities containing query terms
            for term in query_terms:
                if len(term) < 3:  # Skip very short terms
                    continue
                
                cypher_query = """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS $term
                   OR toLower(e.description) CONTAINS $term
                RETURN e.id as id
                LIMIT 10
                """
                
                result = session.run(cypher_query, term=term)
                for record in result:
                    entity_id = record['id']
                    if entity_id not in entity_ids:
                        entity_ids.append(entity_id)
        
        logger.info(f"Entity seed search found {len(entity_ids)} entities for query: {query}")
        return entity_ids
    
    def graph_expansion(
        self,
        entity_ids: List[str],
        hops: int = 2
    ) -> Dict[str, Any]:
        """
        Traverse graph from seed entities to gather context.
        
        Args:
            entity_ids: List of seed entity IDs
            hops: Maximum number of hops to traverse
            
        Returns:
            Dictionary with chunk_ids, entities, and relationships found
        """
        if not entity_ids:
            return {'chunk_ids': [], 'entities': [], 'relationships': []}
        
        chunk_ids_set: Set[str] = set()
        entities_set: Set[str] = set(entity_ids)
        relationships: List[Dict[str, Any]] = []
        
        with self.driver.session(database=self.database) as session:
            # Build entity list for Cypher query
            entity_list = "', '".join(entity_ids)
            
            # Multi-hop traversal query
            query = f"""
            MATCH path = (e:Entity)-[*1..{hops}]-(target)
            WHERE e.id IN ['{entity_list}']
            WITH path, e, target
            WHERE target:Chunk OR target:Entity
            RETURN DISTINCT
                CASE WHEN target:Chunk THEN target.id ELSE NULL END as chunk_id,
                CASE WHEN target:Entity THEN target.id ELSE NULL END as entity_id,
                length(path) as path_length,
                relationships(path) as rels
            ORDER BY path_length
            LIMIT 100
            """
            
            result = session.run(query)
            
            for record in result:
                chunk_id = record['chunk_id']
                entity_id = record['entity_id']
                path_length = record['path_length']
                
                if chunk_id:
                    chunk_ids_set.add(chunk_id)
                if entity_id:
                    entities_set.add(entity_id)
                
                # Extract relationship information
                rels = record['rels']
                if rels:
                    for rel in rels:
                        relationships.append({
                            'type': rel.type,
                            'properties': dict(rel)
                        })
        
        result_dict = {
            'chunk_ids': list(chunk_ids_set),
            'entities': list(entities_set),
            'relationships': relationships[:50]  # Limit relationships
        }
        
        logger.info(
            f"Graph expansion found {len(result_dict['chunk_ids'])} chunks, "
            f"{len(result_dict['entities'])} entities from {len(entity_ids)} seeds"
        )
        
        return result_dict
    
    def get_similar_chunks(
        self,
        chunk_id: str,
        k: int = 5
    ) -> List[str]:
        """
        Get similar chunks via SIMILAR relationships.
        
        Args:
            chunk_id: Source chunk ID
            k: Number of similar chunks to return
            
        Returns:
            List of similar chunk IDs
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (c1:Chunk {id: $chunk_id})-[r:SIMILAR]-(c2:Chunk)
            RETURN c2.id as id, r.similarity_score as score
            ORDER BY score DESC
            LIMIT $k
            """
            
            result = session.run(query, chunk_id=chunk_id, k=k)
            similar_ids = [record['id'] for record in result]
            
            return similar_ids
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session(database=self.database) as session:
            query = """
            MATCH (n)
            RETURN labels(n)[0] as label, count(n) as count
            UNION ALL
            MATCH ()-[r]->()
            RETURN type(r) as label, count(r) as count
            """
            
            result = session.run(query)
            stats = {}
            
            for record in result:
                label = record['label']
                count = record['count']
                stats[label] = count
            
            return stats
    
    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

