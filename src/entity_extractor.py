"""
Entity extraction module for CDER GraphRAG system.
Uses LLM to extract entities and relationships from text chunks.
"""

import json
import hashlib
import os
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document as LangChainDocument

from src.logger import setup_logger

logger = setup_logger(__name__)


class EntityExtractor:
    """
    Extract entities and relationships from text using LLM.
    Handles entity deduplication and normalization.
    """
    
    # Entity types for PDC domain
    ENTITY_TYPES = [
        "ALGORITHM",
        "TECHNOLOGY",
        "CONCEPT",
        "FRAMEWORK",
        "PROTOCOL",
        "SYSTEM",
        "PERSON",
        "ORGANIZATION"
    ]
    
    # Relationship types
    RELATIONSHIP_TYPES = [
        "USES",
        "ENABLES",
        "REQUIRES",
        "CONTRASTS_WITH",
        "EXTENDS",
        "IMPLEMENTS",
        "DEPENDS_ON",
        "SIMILAR_TO"
    ]
    
    def __init__(
        self,
        llm_model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        provider: str = "openai"
    ):
        """
        Initialize entity extractor.
        
        Args:
            llm_model: LLM model name
            api_key: OpenAI API key
            temperature: Sampling temperature (low for structured extraction)
            provider: "openai"
        """
        if not api_key:
            raise ValueError("API key is required")
        
        self.provider = provider.lower()
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=api_key
        )
        logger.info(f"Initialized OpenAI entity extractor with model: {llm_model}")
        
        # Create extraction prompts
        self.entity_prompt = self._create_entity_prompt()
        self.relationship_prompt = self._create_relationship_prompt()
        
        logger.info(f"EntityExtractor initialized: model={llm_model}")
    
    def _create_entity_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for entity extraction."""
        template = """You are an expert in Parallel and Distributed Computing.
Extract key entities from the following text chunk. Focus on technical concepts, algorithms, technologies, and frameworks.

Text chunk:
{text}

Extract entities as a JSON array. Each entity should have:
- name: The entity name (exact as in text)
- type: One of {entity_types}
- description: Brief description (1-2 sentences)

Return ONLY valid JSON array, no other text.
Example format:
[
  {{"name": "Message Passing Interface", "type": "PROTOCOL", "description": "Standard for message passing in parallel computing"}},
  {{"name": "MapReduce", "type": "ALGORITHM", "description": "Programming model for processing large datasets"}}
]

Entities:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _create_relationship_prompt(self) -> ChatPromptTemplate:
        """Create prompt template for relationship extraction."""
        template = """You are an expert in Parallel and Distributed Computing.
Given a list of entities, identify relationships between them in the following text.

Text chunk:
{text}

Entities:
{entities}

Extract relationships as a JSON array. Each relationship should have:
- source: Source entity name
- relationship: One of {relationship_types}
- target: Target entity name
- strength: Confidence score (0.0-1.0)

Return ONLY valid JSON array, no other text.
Example format:
[
  {{"source": "Hadoop", "relationship": "IMPLEMENTS", "target": "MapReduce", "strength": 0.9}},
  {{"source": "MPI", "relationship": "ENABLES", "target": "Distributed Computing", "strength": 0.8}}
]

Relationships:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract entities from text chunk.
        
        Args:
            text: Text chunk to analyze
            
        Returns:
            List of entity dictionaries with name, type, description
        """
        if not text or len(text.strip()) < 50:
            return []
        
        try:
            # Format prompt
            formatted_prompt = self.entity_prompt.format_messages(
                text=text[:2000],  # Limit text length
                entity_types=", ".join(self.ENTITY_TYPES)
            )
            
            # Call LLM
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            entities = json.loads(content)
            
            # Validate entities
            validated_entities = []
            for entity in entities:
                if isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                    # Verify entity appears in text
                    entity_name = entity['name']
                    if entity_name.lower() in text.lower():
                        validated_entities.append({
                            'name': entity_name,
                            'type': entity.get('type', 'CONCEPT'),
                            'description': entity.get('description', '')
                        })
            
            logger.debug(f"Extracted {len(validated_entities)} entities from text")
            return validated_entities
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_relationships(
        self,
        text: str,
        entities: List[Dict[str, str]]
    ) -> List[Dict[str, any]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Text chunk containing entities
            entities: List of extracted entities
            
        Returns:
            List of relationship dictionaries
        """
        if not entities or len(entities) < 2:
            return []
        
        try:
            # Format entity list for prompt
            entity_list = "\n".join([
                f"- {e['name']} ({e['type']})" for e in entities
            ])
            
            # Format prompt
            formatted_prompt = self.relationship_prompt.format_messages(
                text=text[:2000],
                entities=entity_list,
                relationship_types=", ".join(self.RELATIONSHIP_TYPES)
            )
            
            # Call LLM
            response = self.llm.invoke(formatted_prompt)
            content = response.content.strip()
            
            # Parse JSON response
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            relationships = json.loads(content)
            
            # Validate relationships
            validated_rels = []
            entity_names = {e['name'].lower() for e in entities}
            
            for rel in relationships:
                if isinstance(rel, dict) and all(k in rel for k in ['source', 'relationship', 'target']):
                    source = rel['source'].lower()
                    target = rel['target'].lower()
                    
                    # Verify both entities are in the list
                    if source in entity_names and target in entity_names:
                        validated_rels.append({
                            'source': rel['source'],
                            'relationship': rel.get('relationship', 'RELATES_TO'),
                            'target': rel['target'],
                            'strength': float(rel.get('strength', 0.5))
                        })
            
            logger.debug(f"Extracted {len(validated_rels)} relationships")
            return validated_rels
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse relationship JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error extracting relationships: {e}")
            return []
    
    def deduplicate_entities(
        self,
        entities: List[Dict[str, str]],
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, str]]:
        """
        Deduplicate similar entity names.
        
        Args:
            entities: List of entity dictionaries
            similarity_threshold: Similarity threshold for merging (0.0-1.0)
            
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []
        
        # Simple deduplication: exact name matches and case variations
        seen_names = {}
        deduplicated = []
        
        for entity in entities:
            name = entity['name']
            name_lower = name.lower().strip()
            
            # Check for exact match (case-insensitive)
            if name_lower in seen_names:
                # Merge with existing entity
                existing = seen_names[name_lower]
                # Keep longer description
                if len(entity.get('description', '')) > len(existing.get('description', '')):
                    existing['description'] = entity.get('description', '')
                # Update frequency if tracked
                if 'frequency' in existing:
                    existing['frequency'] = existing.get('frequency', 1) + 1
            else:
                # New entity
                entity_copy = entity.copy()
                entity_copy['frequency'] = 1
                seen_names[name_lower] = entity_copy
                deduplicated.append(entity_copy)
        
        # Also check for abbreviations (e.g., "MPI" vs "Message Passing Interface")
        # This is a simplified version - could use embeddings for better matching
        final_entities = []
        for entity in deduplicated:
            name = entity['name']
            # Check if this is an abbreviation of another entity
            is_abbrev = False
            for other in deduplicated:
                if entity == other:
                    continue
                other_name = other['name']
                # Simple check: if one is much shorter and appears in the other
                if len(name) <= 5 and name.upper() in other_name.upper():
                    is_abbrev = True
                    break
            
            if not is_abbrev:
                final_entities.append(entity)
        
        logger.info(f"Deduplicated {len(entities)} entities to {len(final_entities)}")
        return final_entities
    
    def normalize_entity_names(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize entity names to standard format.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            List with normalized entity names
        """
        normalized = []
        
        for entity in entities:
            name = entity['name']
            
            # Normalize: title case, remove extra spaces
            name_normalized = ' '.join(name.split())
            
            # Handle common patterns
            # Acronyms: ensure uppercase
            if len(name_normalized) <= 5 and name_normalized.isupper():
                name_normalized = name_normalized.upper()
            else:
                # Title case for multi-word entities
                words = name_normalized.split()
                if len(words) > 1:
                    name_normalized = ' '.join([w.capitalize() for w in words])
            
            entity_copy = entity.copy()
            entity_copy['name'] = name_normalized
            normalized.append(entity_copy)
        
        return normalized
    
    def batch_extract_entities(
        self,
        chunks: List[LangChainDocument],
        batch_size: int = 5
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract entities from multiple chunks in batches.
        
        Args:
            chunks: List of document chunks
            batch_size: Number of chunks to process per batch
            
        Returns:
            Dictionary mapping chunk_id to list of entities
        """
        results = {}
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            logger.info(f"Processing entity extraction batch {i//batch_size + 1}")
            
            for chunk in batch:
                chunk_id = chunk.metadata.get('chunk_id', f"chunk_{i}")
                text = chunk.page_content
                
                entities = self.extract_entities(text)
                if entities:
                    # Deduplicate and normalize
                    entities = self.deduplicate_entities(entities)
                    entities = self.normalize_entity_names(entities)
                    results[chunk_id] = entities
        
        logger.info(f"Extracted entities from {len(results)} chunks")
        return results

