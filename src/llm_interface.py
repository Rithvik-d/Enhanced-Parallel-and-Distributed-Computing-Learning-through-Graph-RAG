"""
LLM interface for CDER GraphRAG system.
Handles GPT-4 API calls, prompt engineering, and response processing.
"""

import time
import re
from typing import Dict, List, Optional, Any
import logging
from functools import lru_cache

import tiktoken
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.logger import setup_logger

logger = setup_logger(__name__)


class LLMInterface:
    """
    Interface for OpenAI GPT-4 API.
    Handles answer generation, entity extraction, and token management.
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 300,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM interface.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo")
            api_key: OpenAI API key
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
        """
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        
        # Default system prompt
        self.system_prompt = system_prompt or (
            "You are an expert teaching assistant for Parallel and Distributed Computing.\n"
            "Answer questions based ONLY on the provided context from CDER curriculum.\n"
            "If information is insufficient, acknowledge the limitation.\n"
            "Always cite specific document sections in your answer."
        )
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"LLMInterface initialized: model={model}, temperature={temperature}")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def truncate_context(
        self,
        context: str,
        max_tokens: int,
        preserve_end: bool = True
    ) -> str:
        """
        Truncate context to fit token budget.
        
        Args:
            context: Context text to truncate
            max_tokens: Maximum allowed tokens
            preserve_end: If True, keep the end of context; if False, keep the beginning
            
        Returns:
            Truncated context
        """
        tokens = self.encoding.encode(context)
        
        if len(tokens) <= max_tokens:
            return context
        
        if preserve_end:
            # Keep the end (most recent context)
            truncated_tokens = tokens[-max_tokens:]
        else:
            # Keep the beginning
            truncated_tokens = tokens[:max_tokens]
        
        return self.encoding.decode(truncated_tokens)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_answer(
        self,
        query: str,
        context: str,
        retrieval_mode: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Generate answer using GPT-4 with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context from RAG system
            retrieval_mode: Retrieval mode used (for metadata)
            
        Returns:
            Dictionary with answer, tokens_used, latency, citations, confidence
        """
        start_time = time.time()
        
        # Prepare prompt
        user_prompt = self._format_prompt(query, context)
        
        # Count tokens
        system_tokens = self.count_tokens(self.system_prompt)
        user_tokens = self.count_tokens(user_prompt)
        total_input_tokens = system_tokens + user_tokens
        
        # Check token budget
        if total_input_tokens > 8000:  # Approximate limit for GPT-4
            logger.warning(f"Input tokens ({total_input_tokens}) exceed budget, truncating context")
            context = self.truncate_context(context, max_tokens=6000)
            user_prompt = self._format_prompt(query, context)
            user_tokens = self.count_tokens(user_prompt)
            total_input_tokens = system_tokens + user_tokens
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response
            answer = response.choices[0].message.content
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            
            # Extract citations
            citations = self.extract_citations(answer, context)
            
            # Calculate confidence (simple heuristic)
            confidence = self._estimate_confidence(answer, context, query)
            
            latency_ms = (time.time() - start_time) * 1000
            
            result = {
                'answer': answer,
                'tokens_used': {
                    'input': total_input_tokens,
                    'output': output_tokens,
                    'total': total_tokens
                },
                'latency_ms': latency_ms,
                'citations': citations,
                'confidence': confidence,
                'retrieval_mode': retrieval_mode,
                'model': self.model
            }
            
            logger.info(
                f"Generated answer: {total_tokens} tokens, {latency_ms:.0f}ms, "
                f"confidence={confidence:.2f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    def _format_prompt(self, query: str, context: str) -> str:
        """
        Format prompt with context and query.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        return f"""Context:
{context}

Question: {query}

Answer:"""
    
    def extract_citations(self, answer: str, context: str) -> List[str]:
        """
        Extract citations from answer text.
        
        Args:
            answer: Generated answer text
            context: Source context (for validation)
            
        Returns:
            List of citation strings
        """
        citations = []
        
        # Pattern 1: [Source: X, Chapter: Y] format
        pattern1 = r'\[Source:\s*([^,]+),\s*Chapter:\s*([^\]]+)\]'
        matches = re.findall(pattern1, answer)
        for match in matches:
            citations.append(f"Source: {match[0]}, Chapter: {match[1]}")
        
        # Pattern 2: [Chapter X] format
        pattern2 = r'\[Chapter\s+(\d+)\]'
        matches = re.findall(pattern2, answer)
        for match in matches:
            citations.append(f"Chapter {match}")
        
        # Pattern 3: (Source: X) format
        pattern3 = r'\(Source:\s*([^)]+)\)'
        matches = re.findall(pattern3, answer)
        citations.extend([f"Source: {m}" for m in matches])
        
        # Remove duplicates
        citations = list(dict.fromkeys(citations))
        
        return citations
    
    def _estimate_confidence(
        self,
        answer: str,
        context: str,
        query: str
    ) -> float:
        """
        Estimate answer confidence based on heuristics.
        
        Args:
            answer: Generated answer
            context: Source context
            query: Original query
            
        Returns:
            Confidence score (0.0-1.0)
        """
        confidence = 0.5  # Base confidence
        
        # Check if answer mentions uncertainty
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "unclear", "insufficient",
            "not provided", "not available", "cannot determine"
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in uncertainty_phrases):
            confidence -= 0.3
        
        # Check if answer length is reasonable
        if len(answer) < 20:
            confidence -= 0.2
        elif len(answer) > 100:
            confidence += 0.1
        
        # Check if citations are present
        citations = self.extract_citations(answer, context)
        if citations:
            confidence += 0.2
        
        # Check if answer contains query terms
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        overlap = len(query_terms.intersection(answer_terms)) / max(len(query_terms), 1)
        confidence += overlap * 0.1
        
        # Clamp to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract potential entity names from query using simple heuristics.
        
        Args:
            query: User query text
            
        Returns:
            List of potential entity names
        """
        # Simple extraction: capitalized words and technical terms
        words = query.split()
        entities = []
        
        # Find capitalized sequences (potential proper nouns/technical terms)
        current_entity = []
        for word in words:
            if word[0].isupper() and len(word) > 2:
                current_entity.append(word)
            else:
                if len(current_entity) >= 1:
                    entities.append(' '.join(current_entity))
                current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        # Also look for common technical patterns
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms (MPI, GPU, etc.)
            r'\b\w+-\w+\b',  # Hyphenated terms
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates and filter
        entities = list(dict.fromkeys(entities))
        entities = [e for e in entities if len(e) > 2]
        
        return entities
    
    @lru_cache(maxsize=100)
    def _cached_entity_extraction(self, text_hash: str) -> List[str]:
        """
        Cached entity extraction (helper for caching).
        Note: This is a placeholder - actual caching should use text content.
        """
        return []
    
    def close(self) -> None:
        """Close any resources (placeholder for future use)."""
        pass

