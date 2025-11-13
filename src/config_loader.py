"""
Configuration loader for CDER GraphRAG system.
Loads configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


class DocumentConfig(BaseModel):
    """Document processing configuration."""
    source_dir: str
    supported_formats: list[str]
    total_chapters: int


class ProcessingConfig(BaseModel):
    """Text processing configuration."""
    chunk_size: int = Field(gt=0, le=2048)
    chunk_overlap: int = Field(ge=0)
    batch_size: int = Field(gt=0)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: str
    model: str
    dimensions: int = Field(gt=0)
    cache_embeddings: bool = True


class GraphDBConfig(BaseModel):
    """Graph database configuration."""
    provider: str
    aura_enabled: bool = True
    create_indexes: bool = True
    entity_extraction_enabled: bool = True


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    provider: str
    persist_enabled: bool = True
    similarity_threshold: float = Field(ge=0.0, le=1.0)


class VectorRetrievalConfig(BaseModel):
    """Vector retrieval configuration."""
    top_k: int = Field(gt=0)
    similarity_metric: str = "cosine"


class GraphRetrievalConfig(BaseModel):
    """Graph retrieval configuration."""
    max_hops: int = Field(gt=0, le=5)
    include_similar_chunks: bool = True


class HybridRetrievalConfig(BaseModel):
    """Hybrid retrieval configuration."""
    vector_weight: float = Field(ge=0.0, le=1.0)
    graph_weight: float = Field(ge=0.0, le=1.0)
    fusion_strategy: str = "ranked_union"


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    modes: list[str]
    default_mode: str
    vector_config: VectorRetrievalConfig
    graph_config: GraphRetrievalConfig
    hybrid_config: HybridRetrievalConfig


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str
    model: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    top_p: float = Field(ge=0.0, le=1.0)
    system_prompt: str


class ProjectConfig(BaseModel):
    """Complete project configuration."""
    documents: DocumentConfig
    processing: ProcessingConfig
    embeddings: EmbeddingConfig
    graph_database: GraphDBConfig
    vector_database: VectorDBConfig
    retrieval: RetrievalConfig
    llm: LLMConfig


class ConfigLoader:
    """Load and manage configuration from YAML and environment variables."""
    
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        env_path: str = ".env"
    ):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
            env_path: Path to .env file
        """
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self.config: Optional[ProjectConfig] = None
        self.env_vars: Dict[str, Any] = {}
        
    def load(self) -> ProjectConfig:
        """
        Load configuration from YAML and environment variables.
        
        Returns:
            Validated project configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If configuration is invalid
        """
        # Load environment variables
        if self.env_path.exists():
            load_dotenv(self.env_path)
            self._load_env_vars()
        
        # Load YAML configuration
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Override with environment variables
        yaml_config = self._override_with_env(yaml_config)
        
        # Validate and create config model
        try:
            self.config = ProjectConfig(**yaml_config)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")
        
        return self.config
    
    def _load_env_vars(self) -> None:
        """Load environment variables into dictionary."""
        self.env_vars = {
            'NEO4J_URI': os.getenv('NEO4J_URI'),
            'NEO4J_USER': os.getenv('NEO4J_USER'),
            'NEO4J_PASSWORD': os.getenv('NEO4J_PASSWORD'),
            'NEO4J_DATABASE': os.getenv('NEO4J_DATABASE', 'neo4j'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'EMBEDDING_PROVIDER': os.getenv('EMBEDDING_PROVIDER'),
            'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL'),
            'EMBEDDING_DIMENSIONS': os.getenv('EMBEDDING_DIMENSIONS'),
            'CHUNK_SIZE': os.getenv('CHUNK_SIZE'),
            'CHUNK_OVERLAP': os.getenv('CHUNK_OVERLAP'),
            'VECTOR_SEARCH_K': os.getenv('VECTOR_SEARCH_K'),
            'GRAPH_SEARCH_HOPS': os.getenv('GRAPH_SEARCH_HOPS'),
            'HYBRID_VECTOR_WEIGHT': os.getenv('HYBRID_VECTOR_WEIGHT'),
            'HYBRID_GRAPH_WEIGHT': os.getenv('HYBRID_GRAPH_WEIGHT'),
            'LLM_MODEL': os.getenv('LLM_MODEL'),
            'LLM_TEMPERATURE': os.getenv('LLM_TEMPERATURE'),
            'MAX_RESPONSE_TOKENS': os.getenv('MAX_RESPONSE_TOKENS'),
        }
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override YAML config with environment variables.
        
        Args:
            config: YAML configuration dictionary
            
        Returns:
            Updated configuration dictionary
        """
        # Override Neo4j settings
        if self.env_vars.get('NEO4J_URI'):
            # Store in a separate section for runtime access
            if 'runtime' not in config:
                config['runtime'] = {}
            config['runtime']['neo4j_uri'] = self.env_vars['NEO4J_URI']
            config['runtime']['neo4j_user'] = self.env_vars.get('NEO4J_USER', 'neo4j')
            config['runtime']['neo4j_password'] = self.env_vars.get('NEO4J_PASSWORD', '')
            config['runtime']['neo4j_database'] = self.env_vars.get('NEO4J_DATABASE', 'neo4j')
        
        # Override OpenAI API key
        if self.env_vars.get('OPENAI_API_KEY'):
            if 'runtime' not in config:
                config['runtime'] = {}
            config['runtime']['openai_api_key'] = self.env_vars['OPENAI_API_KEY']
        
        # Override embedding settings
        if self.env_vars.get('EMBEDDING_PROVIDER'):
            config['embeddings']['provider'] = self.env_vars['EMBEDDING_PROVIDER']
        if self.env_vars.get('EMBEDDING_MODEL'):
            config['embeddings']['model'] = self.env_vars['EMBEDDING_MODEL']
        if self.env_vars.get('EMBEDDING_DIMENSIONS'):
            config['embeddings']['dimensions'] = int(self.env_vars['EMBEDDING_DIMENSIONS'])
        
        # Override processing settings
        if self.env_vars.get('CHUNK_SIZE'):
            config['processing']['chunk_size'] = int(self.env_vars['CHUNK_SIZE'])
        if self.env_vars.get('CHUNK_OVERLAP'):
            config['processing']['chunk_overlap'] = int(self.env_vars['CHUNK_OVERLAP'])
        
        # Override retrieval settings
        if self.env_vars.get('VECTOR_SEARCH_K'):
            config['retrieval']['vector_config']['top_k'] = int(self.env_vars['VECTOR_SEARCH_K'])
        if self.env_vars.get('GRAPH_SEARCH_HOPS'):
            config['retrieval']['graph_config']['max_hops'] = int(self.env_vars['GRAPH_SEARCH_HOPS'])
        if self.env_vars.get('HYBRID_VECTOR_WEIGHT'):
            config['retrieval']['hybrid_config']['vector_weight'] = float(self.env_vars['HYBRID_VECTOR_WEIGHT'])
        if self.env_vars.get('HYBRID_GRAPH_WEIGHT'):
            config['retrieval']['hybrid_config']['graph_weight'] = float(self.env_vars['HYBRID_GRAPH_WEIGHT'])
        
        # Override LLM settings
        if self.env_vars.get('LLM_MODEL'):
            config['llm']['model'] = self.env_vars['LLM_MODEL']
        if self.env_vars.get('LLM_TEMPERATURE'):
            config['llm']['temperature'] = float(self.env_vars['LLM_TEMPERATURE'])
        if self.env_vars.get('MAX_RESPONSE_TOKENS'):
            config['llm']['max_tokens'] = int(self.env_vars['MAX_RESPONSE_TOKENS'])
        
        return config
    
    def validate_required_env_vars(self) -> Dict[str, bool]:
        """
        Validate that all required environment variables are set.
        
        Returns:
            Dictionary mapping variable names to whether they're set
        """
        required_vars = {
            'NEO4J_URI': self.env_vars.get('NEO4J_URI') is not None,
            'NEO4J_USER': self.env_vars.get('NEO4J_USER') is not None,
            'NEO4J_PASSWORD': self.env_vars.get('NEO4J_PASSWORD') is not None,
            'OPENAI_API_KEY': self.env_vars.get('OPENAI_API_KEY') is not None,
        }
        
        missing = [var for var, is_set in required_vars.items() if not is_set]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return required_vars
    
    def get_runtime_config(self) -> Dict[str, Any]:
        """
        Get runtime configuration (API keys, database URIs).
        
        Returns:
            Dictionary with runtime configuration
        """
        if not self.config:
            raise RuntimeError("Configuration not loaded. Call load() first.")
        
        return {
            'neo4j_uri': self.env_vars.get('NEO4J_URI'),
            'neo4j_user': self.env_vars.get('NEO4J_USER'),
            'neo4j_password': self.env_vars.get('NEO4J_PASSWORD'),
            'neo4j_database': self.env_vars.get('NEO4J_DATABASE', 'neo4j'),
            'openai_api_key': self.env_vars.get('OPENAI_API_KEY'),
        }

