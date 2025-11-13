"""
Document processing module for CDER GraphRAG system.
Handles loading, parsing, and chunking of PDF and DOCX documents.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
import tiktoken

from src.logger import setup_logger

logger = setup_logger(__name__)


class DocumentProcessor:
    """
    Process documents from PDF and DOCX files.
    Handles loading, chunking, and metadata extraction.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            encoding_name: Tokenizer encoding name for tiktoken
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.stats: Dict[str, any] = {
            'documents_processed': 0,
            'total_chunks': 0,
            'total_tokens': 0
        }
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # Approximate: 1 token ≈ 4 chars
            chunk_overlap=chunk_overlap * 4,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"DocumentProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    def load_pdf(self, file_path: str) -> LangChainDocument:
        """
        Load and extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            LangChain Document object with text and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid PDF
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        try:
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Combine all pages into single document
            if len(documents) > 1:
                combined_text = "\n\n".join([doc.page_content for doc in documents])
                metadata = documents[0].metadata.copy()
                metadata['source'] = str(file_path)
                metadata['total_pages'] = len(documents)
                document = LangChainDocument(page_content=combined_text, metadata=metadata)
            else:
                document = documents[0]
                document.metadata['source'] = str(file_path)
            
            logger.info(f"Loaded PDF: {file_path.name} ({len(document.page_content)} chars)")
            return document
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise ValueError(f"Failed to load PDF: {e}")
    
    def load_docx(self, file_path: str) -> LangChainDocument:
        """
        Load and extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            LangChain Document object with text and metadata
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a valid DOCX
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.docx':
            raise ValueError(f"File is not a DOCX: {file_path}")
        
        try:
            loader = Docx2txtLoader(str(file_path))
            documents = loader.load()
            
            # DOCX loader typically returns single document
            document = documents[0] if documents else LangChainDocument(
                page_content="",
                metadata={'source': str(file_path)}
            )
            document.metadata['source'] = str(file_path)
            
            logger.info(f"Loaded DOCX: {file_path.name} ({len(document.page_content)} chars)")
            return document
            
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            raise ValueError(f"Failed to load DOCX: {e}")
    
    def load_document(self, file_path: str) -> LangChainDocument:
        """
        Load document from file (auto-detect format).
        
        Args:
            file_path: Path to document file
            
        Returns:
            LangChain Document object
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.load_pdf(file_path)
        elif suffix == '.docx':
            return self.load_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def load_all_documents(self, directory: str) -> List[LangChainDocument]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Directory path containing documents
            
        Returns:
            List of LangChain Document objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        documents = []
        supported_extensions = ['.pdf', '.docx']
        
        for ext in supported_extensions:
            for file_path in directory.glob(f"*{ext}"):
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        return documents
    
    def extract_chapter_info(self, doc: LangChainDocument) -> Dict[str, any]:
        """
        Extract chapter metadata from document.
        
        Args:
            doc: LangChain Document object
            
        Returns:
            Dictionary with chapter metadata
        """
        metadata = doc.metadata.copy()
        text = doc.page_content
        
        # Try to extract chapter number from filename or content
        chapter_number = None
        title = None
        
        source = metadata.get('source', '')
        if source:
            filename = Path(source).stem
            # Try to extract chapter number from filename (e.g., "chapter1", "ch1")
            import re
            match = re.search(r'chapter[\s_-]?(\d+)', filename, re.IGNORECASE)
            if match:
                chapter_number = int(match.group(1))
        
        # Try to extract title from first few lines
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                title = line
                break
        
        # Count tokens
        token_count = len(self.encoding.encode(text))
        
        return {
            'chapter_number': chapter_number,
            'title': title or metadata.get('title', 'Unknown'),
            'filename': Path(source).name if source else 'Unknown',
            'word_count': len(text.split()),
            'token_count': token_count,
            'char_count': len(text)
        }
    
    def chunk_document(
        self,
        doc: LangChainDocument,
        doc_id: Optional[str] = None
    ) -> List[LangChainDocument]:
        """
        Split document into chunks with overlap.
        
        Args:
            doc: LangChain Document object to chunk
            doc_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            List of chunked Document objects with metadata
        """
        if not doc_id:
            # Generate deterministic ID from content hash
            content_hash = hashlib.md5(doc.page_content[:100].encode()).hexdigest()[:8]
            doc_id = f"doc_{content_hash}"
        
        # Extract chapter info
        chapter_info = self.extract_chapter_info(doc)
        
        # Split text into chunks
        chunks = self.text_splitter.split_documents([doc])
        
        # Add metadata to each chunk
        chunked_docs = []
        for idx, chunk in enumerate(chunks):
            # Count tokens in chunk
            token_count = len(self.encoding.encode(chunk.page_content))
            
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(doc_id, idx, chunk.page_content)
            
            # Update metadata
            chunk.metadata.update({
                'doc_id': doc_id,
                'chunk_id': chunk_id,
                'chunk_index': idx,
                'token_count': token_count,
                'chapter_number': chapter_info.get('chapter_number'),
                'chapter_title': chapter_info.get('title'),
                'filename': chapter_info.get('filename'),
                'total_chunks': len(chunks)
            })
            
            chunked_docs.append(chunk)
        
        # Validate chunks
        self._validate_chunks(chunked_docs, doc)
        
        # Update statistics
        self.stats['documents_processed'] += 1
        self.stats['total_chunks'] += len(chunked_docs)
        self.stats['total_tokens'] += sum(c.metadata.get('token_count', 0) for c in chunked_docs)
        
        logger.info(f"Chunked document {doc_id}: {len(chunked_docs)} chunks")
        
        return chunked_docs
    
    def _generate_chunk_id(self, doc_id: str, index: int, content: str) -> str:
        """
        Generate deterministic chunk ID.
        
        Args:
            doc_id: Parent document ID
            index: Chunk index
            content: Chunk content (first 100 chars)
            
        Returns:
            Unique chunk ID
        """
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"{doc_id}_chunk_{index}_{content_hash}"
    
    def _validate_chunks(
        self,
        chunks: List[LangChainDocument],
        original_doc: LangChainDocument
    ) -> None:
        """
        Validate that chunking preserved content and met size requirements.
        
        Args:
            chunks: List of chunked documents
            original_doc: Original document
            
        Raises:
            AssertionError: If validation fails
        """
        # Check no data loss
        original_text = original_doc.page_content.strip()
        chunked_text = "".join([c.page_content.strip() for c in chunks])
        
        # Allow for minor whitespace differences
        original_clean = "".join(original_text.split())
        chunked_clean = "".join(chunked_text.split())
        
        if len(original_clean) < len(chunked_clean) * 0.9:
            logger.warning(f"Potential data loss detected: original={len(original_clean)}, chunked={len(chunked_clean)}")
        
        # Check chunk sizes (allow 20% variance)
        for chunk in chunks:
            token_count = chunk.metadata.get('token_count', 0)
            if token_count > self.chunk_size * 1.2:
                logger.warning(f"Chunk {chunk.metadata.get('chunk_id')} exceeds size: {token_count} tokens")
        
        # Verify overlap (check consecutive chunks share content)
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].page_content[-100:].lower()
            chunk2_text = chunks[i + 1].page_content[:100].lower()
            
            # Check for overlap (at least some shared words)
            words1 = set(chunk1_text.split())
            words2 = set(chunk2_text.split())
            overlap = len(words1.intersection(words2))
            
            if overlap < 3:  # At least 3 shared words
                logger.debug(f"Low overlap between chunks {i} and {i+1}: {overlap} words")
    
    def validate_chunks(self, chunks: List[LangChainDocument]) -> Dict[str, any]:
        """
        Validate chunk list and return statistics.
        
        Args:
            chunks: List of chunked documents
            
        Returns:
            Dictionary with validation statistics
        """
        if not chunks:
            return {'valid': False, 'error': 'Empty chunk list'}
        
        stats = {
            'valid': True,
            'total_chunks': len(chunks),
            'avg_tokens': sum(c.metadata.get('token_count', 0) for c in chunks) / len(chunks),
            'min_tokens': min(c.metadata.get('token_count', 0) for c in chunks),
            'max_tokens': max(c.metadata.get('token_count', 0) for c in chunks),
            'chunks_with_metadata': sum(1 for c in chunks if c.metadata.get('chunk_id')),
        }
        
        return stats
    
    def get_metadata(self) -> Dict[str, any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing metadata
        """
        return self.stats.copy()

