"""
Query Rewriter for Agentic RAG.

Transforms queries when initial retrieval fails:
- Query reformulation
- Query decomposition
- Query expansion
"""

import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class RewriteResult:
    """Result of query rewriting."""
    original_query: str
    rewritten_query: str
    strategy: str  # "reformulate", "decompose", "expand"
    sub_queries: List[str]  # For decomposition
    reasoning: str


class QueryRewriter:
    """
    Rewrites queries to improve retrieval.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def rewrite(
        self,
        query: str,
        failed_docs: List[str] = None,
        strategy: str = "auto"
    ) -> RewriteResult:
        """
        Rewrite query to improve retrieval.

        Args:
            query: Original query
            failed_docs: Documents that weren't relevant (for context)
            strategy: "reformulate", "decompose", "expand", or "auto"

        Returns:
            RewriteResult with new query
        """
        if strategy == "auto":
            strategy = self._select_strategy(query)

        if strategy == "reformulate":
            return self._reformulate(query, failed_docs)
        elif strategy == "decompose":
            return self._decompose(query)
        elif strategy == "expand":
            return self._expand(query)
        else:
            return self._reformulate(query, failed_docs)

    def _select_strategy(self, query: str) -> str:
        """Select best rewriting strategy based on query."""
        # Complex/compound questions -> decompose
        if " and " in query.lower() or " or " in query.lower():
            return "decompose"

        # Short queries -> expand
        if len(query.split()) <= 4:
            return "expand"

        # Default -> reformulate
        return "reformulate"

    def _reformulate(
        self,
        query: str,
        failed_docs: List[str] = None
    ) -> RewriteResult:
        """Reformulate query to use different terms."""
        context = ""
        if failed_docs:
            context = f"\n\nThese documents were retrieved but weren't relevant:\n{chr(10).join(failed_docs[:2])}"

        prompt = f"""The following query didn't retrieve good results. Reformulate it.

Original query: {query}
{context}

Reformulate the query to:
1. Use different keywords/synonyms
2. Be more specific or more general as needed
3. Focus on the core information need

Respond in JSON:
{{"rewritten_query": "new query", "reasoning": "why this should work better"}}"""

        response = self.llm_client.complete(
            prompt,
            system="You are a search query optimizer. Reformulate queries to improve retrieval.",
            temperature=0.3,
            max_tokens=200
        )

        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group(0))
                return RewriteResult(
                    original_query=query,
                    rewritten_query=result.get("rewritten_query", query),
                    strategy="reformulate",
                    sub_queries=[],
                    reasoning=result.get("reasoning", "")
                )
        except json.JSONDecodeError:
            pass

        # Fallback: simple reformulation
        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            strategy="reformulate",
            sub_queries=[],
            reasoning="Could not reformulate"
        )

    def _decompose(self, query: str) -> RewriteResult:
        """Decompose complex query into simpler sub-queries."""
        prompt = f"""Break down this complex question into simpler sub-questions.

Query: {query}

Create 2-3 focused sub-questions that together answer the original.

Respond in JSON:
{{"sub_queries": ["question 1", "question 2"], "reasoning": "explanation"}}"""

        response = self.llm_client.complete(
            prompt,
            system="You decompose complex questions into simpler parts.",
            temperature=0.3,
            max_tokens=300
        )

        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group(0))
                sub_queries = result.get("sub_queries", [query])
                return RewriteResult(
                    original_query=query,
                    rewritten_query=sub_queries[0] if sub_queries else query,
                    strategy="decompose",
                    sub_queries=sub_queries,
                    reasoning=result.get("reasoning", "")
                )
        except json.JSONDecodeError:
            pass

        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            strategy="decompose",
            sub_queries=[query],
            reasoning="Could not decompose"
        )

    def _expand(self, query: str) -> RewriteResult:
        """Expand query with related terms and context."""
        prompt = f"""Expand this query with more context and related terms.

Query: {query}

Add relevant context, synonyms, or related concepts.
Keep it as a natural question.

Respond in JSON:
{{"expanded_query": "longer query with more context", "reasoning": "what was added"}}"""

        response = self.llm_client.complete(
            prompt,
            system="You expand short queries with helpful context.",
            temperature=0.3,
            max_tokens=200
        )

        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group(0))
                return RewriteResult(
                    original_query=query,
                    rewritten_query=result.get("expanded_query", query),
                    strategy="expand",
                    sub_queries=[],
                    reasoning=result.get("reasoning", "")
                )
        except json.JSONDecodeError:
            pass

        return RewriteResult(
            original_query=query,
            rewritten_query=query,
            strategy="expand",
            sub_queries=[],
            reasoning="Could not expand"
        )

    def generate_alternatives(self, query: str, n: int = 3) -> List[str]:
        """Generate alternative phrasings of the query."""
        prompt = f"""Generate {n} alternative ways to ask this question.
Use different words and phrasings but keep the same meaning.

Query: {query}

Respond in JSON:
{{"alternatives": ["alt 1", "alt 2", "alt 3"]}}"""

        response = self.llm_client.complete(
            prompt,
            system="You generate alternative phrasings of questions.",
            temperature=0.5,
            max_tokens=300
        )

        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group(0))
                return result.get("alternatives", [])
        except json.JSONDecodeError:
            pass

        return []


class HypotheticalDocumentRewriter:
    """
    HyDE (Hypothetical Document Embeddings) approach.
    Generate a hypothetical answer, then use it for retrieval.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = f"""Imagine a document that perfectly answers this question.
Write a brief passage (2-3 sentences) that would contain the answer.

Question: {query}

Write the hypothetical passage:"""

        response = self.llm_client.complete(
            prompt,
            system="You write hypothetical document passages for retrieval.",
            temperature=0.3,
            max_tokens=200
        )

        return response.content.strip()

    def rewrite(self, query: str) -> RewriteResult:
        """Generate HyDE query."""
        hypothetical = self.generate_hypothetical_doc(query)

        return RewriteResult(
            original_query=query,
            rewritten_query=hypothetical,
            strategy="hyde",
            sub_queries=[],
            reasoning="Using hypothetical document for retrieval"
        )
