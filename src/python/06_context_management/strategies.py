"""
Context Management Strategies

This module implements three approaches for handling long documents:
1. Sliding Window - Keep only recent context within token limit
2. Summarization - Compress earlier context into summaries
3. Hierarchical - Combine summary + recent detail

Each strategy aims to answer questions about documents that exceed
the context window while maintaining accuracy.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable


@dataclass
class ContextResult:
    """Result of context management processing."""
    strategy: str
    original_tokens: int
    final_tokens: int
    compression_ratio: float
    context_used: str
    chunks_included: int
    summary_tokens: int = 0


class ContextStrategy(ABC):
    """Base class for context management strategies."""

    @abstractmethod
    def prepare_context(
        self,
        document: str,
        question: str,
        max_tokens: int = 4000
    ) -> ContextResult:
        """Prepare context for LLM within token limits."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (1 token ≈ 4 characters)."""
        return len(text) // 4

    def chunk_document(self, document: str, chunk_size: int = 500) -> List[str]:
        """Split document into chunks by paragraphs or sections."""
        # Split by chapter headers first
        chapters = re.split(r'\n## ', document)

        chunks = []
        for chapter in chapters:
            if not chapter.strip():
                continue

            # Add header back if it was removed
            if not chapter.startswith('#'):
                chapter = '## ' + chapter

            # If chapter is small enough, keep as one chunk
            if self.estimate_tokens(chapter) <= chunk_size:
                chunks.append(chapter)
            else:
                # Split chapter into paragraphs
                paragraphs = chapter.split('\n\n')
                current_chunk = ""

                for para in paragraphs:
                    if self.estimate_tokens(current_chunk + para) <= chunk_size:
                        current_chunk += para + "\n\n"
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"

                if current_chunk:
                    chunks.append(current_chunk.strip())

        return chunks


class SlidingWindowStrategy(ContextStrategy):
    """
    Sliding Window Context Strategy

    Keeps only the most recent chunks that fit within the token limit.
    Simple but may miss relevant information from earlier sections.
    """

    def __init__(self, chunk_size: int = 500):
        self.chunk_size = chunk_size

    def prepare_context(
        self,
        document: str,
        question: str,
        max_tokens: int = 4000
    ) -> ContextResult:
        """Select chunks from end of document that fit in context."""
        original_tokens = self.estimate_tokens(document)
        chunks = self.chunk_document(document, self.chunk_size)

        # Reserve tokens for question and response
        available_tokens = max_tokens - self.estimate_tokens(question) - 500

        # Add chunks from the end until we hit the limit
        selected_chunks = []
        current_tokens = 0

        for chunk in reversed(chunks):
            chunk_tokens = self.estimate_tokens(chunk)
            if current_tokens + chunk_tokens <= available_tokens:
                selected_chunks.insert(0, chunk)
                current_tokens += chunk_tokens
            else:
                break

        context = "\n\n".join(selected_chunks)

        return ContextResult(
            strategy="sliding_window",
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            compression_ratio=current_tokens / original_tokens if original_tokens > 0 else 1.0,
            context_used=context,
            chunks_included=len(selected_chunks)
        )


class SummarizationStrategy(ContextStrategy):
    """
    Summarization Context Strategy

    Summarizes the entire document to fit within token limits.
    Good compression but may lose specific details.
    """

    def __init__(self, summarize_fn: Callable[[str], str] = None, chunk_size: int = 500):
        self.summarize_fn = summarize_fn
        self.chunk_size = chunk_size

    def prepare_context(
        self,
        document: str,
        question: str,
        max_tokens: int = 4000
    ) -> ContextResult:
        """Summarize document sections to fit in context."""
        original_tokens = self.estimate_tokens(document)
        chunks = self.chunk_document(document, self.chunk_size)

        # Reserve tokens for question and response
        available_tokens = max_tokens - self.estimate_tokens(question) - 500

        # Summarize each chunk
        summaries = []
        total_summary_tokens = 0

        if self.summarize_fn:
            for chunk in chunks:
                # Only summarize if chunk is large enough
                if self.estimate_tokens(chunk) > 100:
                    summary = self.summarize_fn(chunk)
                    summaries.append(summary)
                    total_summary_tokens += self.estimate_tokens(summary)
                else:
                    summaries.append(chunk)
                    total_summary_tokens += self.estimate_tokens(chunk)
        else:
            # Fallback: extract key sentences (first and last of each section)
            for chunk in chunks:
                lines = [l for l in chunk.split('\n') if l.strip()]
                if len(lines) <= 3:
                    summaries.append(chunk)
                else:
                    # Keep header + first 2 lines + last line
                    key_lines = lines[:3] + lines[-1:]
                    summaries.append('\n'.join(key_lines))
                total_summary_tokens = sum(self.estimate_tokens(s) for s in summaries)

        # Combine summaries that fit
        context_parts = []
        current_tokens = 0

        for summary in summaries:
            summary_tokens = self.estimate_tokens(summary)
            if current_tokens + summary_tokens <= available_tokens:
                context_parts.append(summary)
                current_tokens += summary_tokens

        context = "\n\n---\n\n".join(context_parts)

        return ContextResult(
            strategy="summarization",
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            compression_ratio=current_tokens / original_tokens if original_tokens > 0 else 1.0,
            context_used=context,
            chunks_included=len(context_parts),
            summary_tokens=total_summary_tokens
        )


class HierarchicalStrategy(ContextStrategy):
    """
    Hierarchical Context Strategy

    Combines a summary of the full document with detailed recent sections.
    Best of both worlds: global context + specific details.
    """

    def __init__(
        self,
        summarize_fn: Callable[[str], str] = None,
        summary_ratio: float = 0.3,
        chunk_size: int = 500
    ):
        self.summarize_fn = summarize_fn
        self.summary_ratio = summary_ratio  # Portion of tokens for summary
        self.chunk_size = chunk_size

    def prepare_context(
        self,
        document: str,
        question: str,
        max_tokens: int = 4000
    ) -> ContextResult:
        """Create hierarchical context with summary + recent detail."""
        original_tokens = self.estimate_tokens(document)
        chunks = self.chunk_document(document, self.chunk_size)

        # Reserve tokens for question and response
        available_tokens = max_tokens - self.estimate_tokens(question) - 500

        # Allocate tokens
        summary_budget = int(available_tokens * self.summary_ratio)
        detail_budget = available_tokens - summary_budget

        # Create summary of earlier chunks
        summary_parts = []
        summary_tokens = 0

        if self.summarize_fn:
            # Summarize first 2/3 of document
            early_chunks = chunks[:len(chunks) * 2 // 3]
            for chunk in early_chunks:
                if summary_tokens >= summary_budget:
                    break
                summary = self.summarize_fn(chunk)
                summary_parts.append(summary)
                summary_tokens += self.estimate_tokens(summary)
        else:
            # Fallback: extract headers and key sentences
            early_chunks = chunks[:len(chunks) * 2 // 3]
            for chunk in early_chunks:
                if summary_tokens >= summary_budget:
                    break
                lines = [l for l in chunk.split('\n') if l.strip()]
                # Keep just headers and first line of content
                key_lines = [l for l in lines if l.startswith('#')] + lines[:2]
                summary = '\n'.join(key_lines[:3])
                summary_parts.append(summary)
                summary_tokens += self.estimate_tokens(summary)

        # Get recent chunks in full detail
        detail_parts = []
        detail_tokens = 0
        late_chunks = chunks[len(chunks) * 2 // 3:]

        for chunk in late_chunks:
            chunk_tokens = self.estimate_tokens(chunk)
            if detail_tokens + chunk_tokens <= detail_budget:
                detail_parts.append(chunk)
                detail_tokens += chunk_tokens

        # Combine: summary first, then details
        summary_section = "## Document Summary (Earlier Sections)\n\n" + "\n".join(summary_parts)
        detail_section = "## Detailed Content (Recent Sections)\n\n" + "\n\n".join(detail_parts)

        context = summary_section + "\n\n---\n\n" + detail_section
        final_tokens = summary_tokens + detail_tokens

        return ContextResult(
            strategy="hierarchical",
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            compression_ratio=final_tokens / original_tokens if original_tokens > 0 else 1.0,
            context_used=context,
            chunks_included=len(summary_parts) + len(detail_parts),
            summary_tokens=summary_tokens
        )


class RetrievalAugmentedStrategy(ContextStrategy):
    """
    Retrieval-Augmented Context Strategy

    Uses semantic search to find the most relevant chunks for the question.
    Most efficient for specific queries.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]] = None,
        top_k: int = 5,
        chunk_size: int = 500
    ):
        self.embed_fn = embed_fn
        self.top_k = top_k
        self.chunk_size = chunk_size
        self._embeddings_cache = {}

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a * norm_b > 0 else 0

    def prepare_context(
        self,
        document: str,
        question: str,
        max_tokens: int = 4000
    ) -> ContextResult:
        """Retrieve most relevant chunks for the question."""
        original_tokens = self.estimate_tokens(document)
        chunks = self.chunk_document(document, self.chunk_size)

        # Reserve tokens for question and response
        available_tokens = max_tokens - self.estimate_tokens(question) - 500

        if self.embed_fn:
            # Get embeddings for question and chunks
            question_embedding = self.embed_fn(question)

            chunk_scores = []
            for i, chunk in enumerate(chunks):
                # Cache chunk embeddings
                if i not in self._embeddings_cache:
                    self._embeddings_cache[i] = self.embed_fn(chunk)
                chunk_embedding = self._embeddings_cache[i]
                score = self.cosine_similarity(question_embedding, chunk_embedding)
                chunk_scores.append((score, i, chunk))

            # Sort by relevance
            chunk_scores.sort(reverse=True)

            # Select top-k chunks that fit
            selected = []
            current_tokens = 0

            for score, idx, chunk in chunk_scores[:self.top_k * 2]:
                chunk_tokens = self.estimate_tokens(chunk)
                if current_tokens + chunk_tokens <= available_tokens:
                    selected.append((idx, chunk))
                    current_tokens += chunk_tokens

            # Sort by original order for coherence
            selected.sort(key=lambda x: x[0])
            context = "\n\n---\n\n".join(chunk for _, chunk in selected)

        else:
            # Fallback: keyword matching
            question_words = set(question.lower().split())

            chunk_scores = []
            for i, chunk in enumerate(chunks):
                chunk_words = set(chunk.lower().split())
                overlap = len(question_words & chunk_words)
                chunk_scores.append((overlap, i, chunk))

            chunk_scores.sort(reverse=True)

            selected = []
            current_tokens = 0

            for score, idx, chunk in chunk_scores[:self.top_k * 2]:
                chunk_tokens = self.estimate_tokens(chunk)
                if current_tokens + chunk_tokens <= available_tokens:
                    selected.append((idx, chunk))
                    current_tokens += chunk_tokens

            selected.sort(key=lambda x: x[0])
            context = "\n\n---\n\n".join(chunk for _, chunk in selected)
            current_tokens = self.estimate_tokens(context)

        return ContextResult(
            strategy="retrieval_augmented",
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            compression_ratio=current_tokens / original_tokens if original_tokens > 0 else 1.0,
            context_used=context,
            chunks_included=len(selected)
        )


def create_summarizer(llm_client) -> Callable[[str], str]:
    """Create a summarization function using the LLM client."""
    def summarize(text: str) -> str:
        prompt = f"""Summarize the following text in 2-3 sentences, preserving key facts and numbers:

{text}

Summary:"""
        response = llm_client.complete(prompt, temperature=0.0, max_tokens=150)
        return response.content.strip()

    return summarize


def create_embedder():
    """Create an embedding function using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')

        def embed(text: str) -> List[float]:
            return model.encode(text).tolist()

        return embed
    except ImportError:
        return None
