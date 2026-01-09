"""
Enhanced Retriever for Agentic RAG.

Provides document retrieval with:
- Relevance scoring
- Multiple retrieval strategies
- Configurable parameters
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_client import LLMResponse


@dataclass
class RetrievedDocument:
    """A retrieved document with relevance information."""
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0  # Set by evaluator
    relevance_label: str = ""  # "relevant", "partial", "irrelevant"


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""
    query: str
    documents: List[RetrievedDocument]
    retrieval_time_ms: float = 0
    strategy: str = "semantic"

    @property
    def avg_similarity(self) -> float:
        if not self.documents:
            return 0.0
        return sum(d.similarity for d in self.documents) / len(self.documents)

    @property
    def avg_relevance(self) -> float:
        if not self.documents:
            return 0.0
        return sum(d.relevance_score for d in self.documents) / len(self.documents)

    @property
    def has_relevant_docs(self) -> bool:
        return any(d.relevance_label == "relevant" for d in self.documents)


class Retriever:
    """
    Enhanced document retriever with multiple strategies.
    """

    def __init__(self, vectorstore, n_results: int = 3):
        """
        Initialize retriever.

        Args:
            vectorstore: VectorStore instance from 04_rag module
            n_results: Default number of results to retrieve
        """
        self.vectorstore = vectorstore
        self.n_results = n_results

    def retrieve(
        self,
        query: str,
        n_results: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query
            n_results: Number of results (defaults to self.n_results)
            filter_metadata: Optional metadata filter

        Returns:
            RetrievalResult with documents
        """
        import time
        start = time.perf_counter()

        n = n_results or self.n_results
        results = self.vectorstore.search(query, n_results=n, filter_metadata=filter_metadata)

        documents = [
            RetrievedDocument(
                id=r["id"],
                content=r["content"],
                similarity=r["similarity"],
                metadata=r["metadata"],
            )
            for r in results
        ]

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            documents=documents,
            retrieval_time_ms=elapsed,
            strategy="semantic"
        )

    def retrieve_with_expansion(
        self,
        query: str,
        expanded_queries: List[str],
        n_results: int = None
    ) -> RetrievalResult:
        """
        Retrieve using query expansion (multiple related queries).

        Combines results from original + expanded queries, deduplicating.
        """
        import time
        start = time.perf_counter()

        n = n_results or self.n_results
        all_queries = [query] + expanded_queries
        seen_ids = set()
        documents = []

        for q in all_queries:
            results = self.vectorstore.search(q, n_results=n)
            for r in results:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    documents.append(RetrievedDocument(
                        id=r["id"],
                        content=r["content"],
                        similarity=r["similarity"],
                        metadata=r["metadata"],
                    ))

        # Sort by similarity and take top n
        documents.sort(key=lambda d: d.similarity, reverse=True)
        documents = documents[:n * 2]  # Keep more since we expanded

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            documents=documents,
            retrieval_time_ms=elapsed,
            strategy="expansion"
        )

    def retrieve_hybrid(
        self,
        query: str,
        keywords: List[str] = None,
        n_results: int = None
    ) -> RetrievalResult:
        """
        Hybrid retrieval: semantic + keyword matching.

        Boosts results that contain query keywords.
        """
        import time
        start = time.perf_counter()

        n = n_results or self.n_results
        results = self.vectorstore.search(query, n_results=n * 2)

        # Extract keywords from query if not provided
        if keywords is None:
            keywords = self._extract_keywords(query)

        documents = []
        for r in results:
            doc = RetrievedDocument(
                id=r["id"],
                content=r["content"],
                similarity=r["similarity"],
                metadata=r["metadata"],
            )

            # Boost score for keyword matches
            keyword_boost = self._calculate_keyword_boost(r["content"], keywords)
            doc.similarity = min(1.0, doc.similarity + keyword_boost * 0.2)

            documents.append(doc)

        # Re-sort by boosted similarity
        documents.sort(key=lambda d: d.similarity, reverse=True)
        documents = documents[:n]

        elapsed = (time.perf_counter() - start) * 1000

        return RetrievalResult(
            query=query,
            documents=documents,
            retrieval_time_ms=elapsed,
            strategy="hybrid"
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query (simple approach)."""
        import re
        # Remove common words
        stopwords = {"what", "is", "the", "a", "an", "how", "many", "much", "do", "does",
                    "at", "in", "on", "for", "to", "of", "and", "or", "are", "was", "were"}
        words = re.findall(r'\b\w+\b', query.lower())
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _calculate_keyword_boost(self, content: str, keywords: List[str]) -> float:
        """Calculate boost based on keyword presence."""
        if not keywords:
            return 0.0
        content_lower = content.lower()
        matches = sum(1 for kw in keywords if kw in content_lower)
        return matches / len(keywords)
