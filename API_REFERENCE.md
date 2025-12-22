# API Reference

Complete API documentation for the CDER GraphRAG System.

## Table of Contents

- [CDERChatbot](#cderchatbot-class)
- [Retrievers](#retrievers)
- [LLMInterface](#llminterface-class)
- [VectorDBManager](#vectordbmanager-class)
- [GraphDBManager](#graphdbmanager-class)

---

## CDERChatbot Class

Main orchestrator for the RAG system.

### Initialization

```python
from src.chatbot import CDERChatbot

chatbot = CDERChatbot(
    config_path='config/config.yaml',
    env_path='.env'
)
```

**Parameters:**
- `config_path` (str, optional): Path to configuration YAML file. Default: `'config/config.yaml'`
- `env_path` (str, optional): Path to environment variables file. Default: `'.env'`

### Methods

#### `process_query(user_query: str, retrieval_mode: str = "hybrid") -> Dict[str, Any]`

Process a single query through the specified retrieval mode.

**Parameters:**
- `user_query` (str): User's question
- `retrieval_mode` (str): One of `"no-rag"`, `"vector-only"`, `"graph-only"`, `"hybrid"`. Default: `"hybrid"`

**Returns:**
```python
{
    'query': str,                    # Original query
    'answer': str,                   # Generated answer
    'retrieval_mode': str,           # Mode used
    'metadata': {
        'retrieval': {
            'latency_ms': float,     # Retrieval time in milliseconds
            'chunks_retrieved': int, # Number of chunks retrieved
            # ... mode-specific metadata
        },
        'generation': {
            'tokens_used': {
                'input': int,
                'output': int,
                'total': int
            },
            'latency_ms': float,     # Generation time
            'confidence': float      # Confidence score (0.0-1.0)
        },
        'citations': list            # List of citation strings
    }
}
```

**Example:**
```python
result = chatbot.process_query(
    "What is parallel computing?",
    retrieval_mode="hybrid"
)
print(result['answer'])
print(f"Confidence: {result['metadata']['generation']['confidence']}")
```

#### `compare_all_retrievals(user_query: str) -> Dict[str, Any]`

Run query through all four retrieval approaches.

**Parameters:**
- `user_query` (str): User's question

**Returns:**
```python
{
    'no-rag': {...},      # Result from no-rag mode
    'vector-only': {...}, # Result from vector-only mode
    'graph-only': {...}, # Result from graph-only mode
    'hybrid': {...}       # Result from hybrid mode
}
```

**Example:**
```python
results = chatbot.compare_all_retrievals("What is MapReduce?")
for mode, result in results.items():
    print(f"{mode}: {result['answer'][:100]}...")
```

#### `interactive_chat() -> None`

Start interactive command-line chat interface.

**Example:**
```python
chatbot.interactive_chat()
# Commands: /compare, /stats, /history, /exit
```

#### `format_response(result: Dict[str, Any]) -> str`

Format response for display.

**Parameters:**
- `result` (Dict): Result dictionary from `process_query()`

**Returns:** Formatted string

#### `close() -> None`

Close connections and cleanup resources.

---

## Retrievers

### NoRAGRetriever

Baseline retriever that returns empty context.

```python
from src.retrievers import NoRAGRetriever

retriever = NoRAGRetriever()
context = retriever.retrieve("query")
metadata = retriever.get_metadata()
```

**Methods:**
- `retrieve(query: str) -> str`: Returns empty string
- `get_metadata() -> Dict`: Returns metadata with `latency_ms`, `chunks_retrieved`

### VectorRetriever

Semantic similarity search using ChromaDB.

```python
from src.retrievers import VectorRetriever
from src.vector_db import VectorDBManager

vector_manager = VectorDBManager(...)
retriever = VectorRetriever(vector_manager, k=3)
context = retriever.retrieve("query")
```

**Parameters:**
- `vector_manager` (VectorDBManager): Vector database manager
- `k` (int): Number of top results to retrieve. Default: 3

**Methods:**
- `retrieve(query: str) -> str`: Returns concatenated context from top-k chunks
- `get_metadata() -> Dict`: Returns metadata with retrieval stats

### GraphRetriever

Entity and relationship-based graph traversal.

```python
from src.retrievers import GraphRetriever
from src.graph_db import GraphDBManager
from src.vector_db import VectorDBManager
from src.llm_interface import LLMInterface

retriever = GraphRetriever(
    graph_manager=graph_manager,
    vector_manager=vector_manager,
    llm_interface=llm_interface,
    hops=2
)
context = retriever.retrieve("query")
```

**Parameters:**
- `graph_manager` (GraphDBManager): Neo4j graph manager
- `vector_manager` (VectorDBManager): Vector database manager
- `llm_interface` (LLMInterface): LLM interface for entity extraction
- `hops` (int): Maximum graph traversal depth. Default: 2

**Methods:**
- `retrieve(query: str) -> str`: Returns context from graph traversal
- `get_metadata() -> Dict`: Returns metadata with graph stats

### HybridRetriever

Combines vector and graph retrieval.

```python
from src.retrievers import HybridRetriever

retriever = HybridRetriever(
    vector_manager=vector_manager,
    graph_manager=graph_manager,
    llm_interface=llm_interface,
    vector_weight=0.6,
    graph_weight=0.4,
    k=3,
    hops=2,
    fusion_strategy="ranked_union"
)
context = retriever.retrieve("query")
```

**Parameters:**
- `vector_manager` (VectorDBManager): Vector database manager
- `graph_manager` (GraphDBManager): Neo4j graph manager
- `llm_interface` (LLMInterface): LLM interface
- `vector_weight` (float): Weight for vector results (0.0-1.0). Default: 0.6
- `graph_weight` (float): Weight for graph results (0.0-1.0). Default: 0.4
- `k` (int): Number of vector results. Default: 3
- `hops` (int): Graph traversal depth. Default: 2
- `fusion_strategy` (str): `"ranked_union"` or `"weighted_sum"`. Default: `"ranked_union"`

**Methods:**
- `retrieve(query: str) -> str`: Returns fused context
- `get_metadata() -> Dict`: Returns combined metadata

---

## LLMInterface Class

Interface for OpenAI API calls.

### Initialization

```python
from src.llm_interface import LLMInterface

llm = LLMInterface(
    model="gpt-3.5-turbo",
    api_key="sk-proj-...",
    temperature=0.3,
    max_tokens=300,
    provider="openai"
)
```

**Parameters:**
- `model` (str): Model name. Default: `"gpt-4"`
- `api_key` (str): OpenAI API key. Required
- `temperature` (float): Sampling temperature (0.0-2.0). Default: 0.3
- `max_tokens` (int): Maximum tokens in response. Default: 300
- `system_prompt` (str, optional): Custom system prompt
- `provider` (str): LLM provider. Default: `"openai"`

### Methods

#### `generate_answer(query: str, context: str, retrieval_mode: str = "hybrid") -> Dict[str, Any]`

Generate answer using LLM with retrieved context.

**Parameters:**
- `query` (str): User query
- `context` (str): Retrieved context from RAG system
- `retrieval_mode` (str): Retrieval mode used. Default: `"hybrid"`

**Returns:**
```python
{
    'answer': str,
    'tokens_used': {
        'input': int,
        'output': int,
        'total': int
    },
    'latency_ms': float,
    'citations': list,
    'confidence': float,
    'retrieval_mode': str,
    'model': str
}
```

#### `extract_entities_from_query(query: str) -> List[str]`

Extract potential entity names from query using heuristics.

**Parameters:**
- `query` (str): User query text

**Returns:** List of potential entity names

#### `count_tokens(text: str) -> int`

Count tokens in text.

**Parameters:**
- `text` (str): Input text

**Returns:** Number of tokens

#### `truncate_context(context: str, max_tokens: int, preserve_end: bool = True) -> str`

Truncate context to fit token budget.

**Parameters:**
- `context` (str): Context text to truncate
- `max_tokens` (int): Maximum allowed tokens
- `preserve_end` (bool): If True, keep the end; if False, keep the beginning. Default: True

**Returns:** Truncated context

---

## VectorDBManager Class

Manages ChromaDB vector database operations.

### Initialization

```python
from src.vector_db import VectorDBManager

vector_db = VectorDBManager(
    collection_name="cder_embeddings",
    persist_dir="./artifacts/vector_store",
    embedding_provider="sentence-transformers",
    embedding_model="all-MiniLM-L6-v2"
)
```

**Parameters:**
- `collection_name` (str): Collection name. Default: `"cder_embeddings"`
- `persist_dir` (str): Directory to persist database. Default: `"./artifacts/vector_store"`
- `embedding_provider` (str): `"sentence-transformers"` or `"openai"`. Default: `"sentence-transformers"`
- `embedding_model` (str): Model name. Default: `"all-MiniLM-L6-v2"`
- `api_key` (str, optional): API key for OpenAI embeddings

### Methods

#### `add_documents(documents: List[LangChainDocument]) -> List[str]`

Add documents to vector store.

**Parameters:**
- `documents` (List[LangChainDocument]): List of documents

**Returns:** List of document IDs

#### `similarity_search(query: str, k: int = 5) -> List[Dict]`

Search for similar documents.

**Parameters:**
- `query` (str): Query text
- `k` (int): Number of results. Default: 5

**Returns:** List of result dictionaries with `text`, `metadata`, `distance`

#### `get_by_id(chunk_id: str) -> Optional[Dict]`

Get document by ID.

**Parameters:**
- `chunk_id` (str): Chunk ID

**Returns:** Document dictionary or None

---

## GraphDBManager Class

Manages Neo4j graph database operations.

### Initialization

```python
from src.graph_db import GraphDBManager

graph_db = GraphDBManager(
    uri="neo4j+s://...",
    user="neo4j",
    password="password",
    database="neo4j"
)
```

**Parameters:**
- `uri` (str): Neo4j connection URI
- `user` (str): Username
- `password` (str): Password
- `database` (str): Database name. Default: `"neo4j"`

### Methods

#### `entity_seed_search(query: str) -> List[str]`

Find entity nodes matching query terms.

**Parameters:**
- `query` (str): Query text

**Returns:** List of entity IDs

#### `graph_expansion(entity_ids: List[str], hops: int = 2) -> Dict[str, Any]`

Traverse graph from seed entities.

**Parameters:**
- `entity_ids` (List[str]): List of seed entity IDs
- `hops` (int): Maximum hops. Default: 2

**Returns:**
```python
{
    'chunk_ids': List[str],
    'entities': List[str],
    'relationships': List[Dict]
}
```

#### `add_document_node(doc_id: str, metadata: Dict) -> str`

Add document node to graph.

#### `add_chunk_nodes(chunks: List[LangChainDocument], parent_doc_id: str) -> List[str]`

Add chunk nodes to graph.

#### `add_entity_node(name: str, entity_type: str, description: Optional[str] = None) -> str`

Add entity node to graph.

---

## Error Handling

All classes raise appropriate exceptions:

- `ValueError`: Invalid parameters
- `ConnectionError`: Database connection failures
- `RateLimitError`: API rate limit exceeded (auto-retries)
- `APIError`: General API errors

---

**Last Updated**: November 2025



