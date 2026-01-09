"""
Response Synthesizer for Agentic RAG.

Generates final responses with:
- Citation support
- Confidence scoring
- Fallback handling
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from retriever import RetrievedDocument, RetrievalResult


@dataclass
class Citation:
    """A citation to source document."""
    doc_id: str
    text: str  # Quoted text from source
    relevance: float


@dataclass
class SynthesizedResponse:
    """Response with citations and confidence."""
    answer: str
    confidence: float  # 0.0 to 1.0
    citations: List[Citation]
    sources_used: int
    is_fallback: bool = False
    fallback_reason: str = ""
    tokens_used: int = 0
    latency_ms: float = 0


class ResponseSynthesizer:
    """
    Synthesizes responses from retrieved documents.
    """

    def __init__(self, llm_client, include_citations: bool = True):
        self.llm_client = llm_client
        self.include_citations = include_citations

    def synthesize(
        self,
        query: str,
        documents: List[RetrievedDocument],
        context_type: str = "general"
    ) -> SynthesizedResponse:
        """
        Generate response from retrieved documents.

        Args:
            query: The user's question
            documents: Retrieved relevant documents
            context_type: "general", "factual", "technical"

        Returns:
            SynthesizedResponse with answer and citations
        """
        import time
        start = time.perf_counter()

        # Filter to relevant documents (use similarity if relevance not set)
        relevant_docs = [
            d for d in documents
            if d.relevance_score >= 0.4 or (d.relevance_score == 0 and d.similarity >= 0.3)
        ]

        if not relevant_docs:
            return self._generate_fallback(query, "No relevant documents found")

        # Build context
        context = self._build_context(relevant_docs)

        # Generate response
        prompt = self._build_prompt(query, context, context_type)

        response = self.llm_client.complete(
            prompt,
            system=self._get_system_prompt(context_type),
            temperature=0.2,
            max_tokens=800
        )

        elapsed = (time.perf_counter() - start) * 1000

        # Parse response
        answer, citations, confidence = self._parse_response(response.content, relevant_docs)

        return SynthesizedResponse(
            answer=answer,
            confidence=confidence,
            citations=citations,
            sources_used=len(relevant_docs),
            tokens_used=response.total_tokens,
            latency_ms=elapsed
        )

    def _build_context(self, documents: List[RetrievedDocument]) -> str:
        """Build context string from documents."""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[Source {i}] (relevance: {doc.relevance_score:.0%})\n{doc.content}")
        return "\n\n".join(parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        context_type: str
    ) -> str:
        """Build the generation prompt."""
        citation_instruction = ""
        if self.include_citations:
            citation_instruction = """
Include citations using [Source N] format when referencing information.
"""

        return f"""Based on the following sources, answer the question.
{citation_instruction}
## Sources
{context}

## Question
{query}

## Instructions
- Answer based ONLY on the provided sources
- If the sources don't contain the answer, say so
- Be concise but complete
- Rate your confidence (0.0-1.0) based on source relevance

Respond in this format:
ANSWER: [your answer with [Source N] citations]
CONFIDENCE: [0.0-1.0]"""

    def _get_system_prompt(self, context_type: str) -> str:
        """Get system prompt based on context type."""
        prompts = {
            "general": "You answer questions based on provided sources. Be accurate and cite your sources.",
            "factual": "You are a fact-focused assistant. Only state what is explicitly in the sources. No assumptions.",
            "technical": "You are a technical expert. Provide precise, technical answers based on documentation."
        }
        return prompts.get(context_type, prompts["general"])

    def _parse_response(
        self,
        response: str,
        documents: List[RetrievedDocument]
    ) -> tuple:
        """Parse response into answer, citations, and confidence."""
        # Extract answer
        answer_match = re.search(r'ANSWER:\s*(.*?)(?=CONFIDENCE:|$)', response, re.DOTALL)
        answer = answer_match.group(1).strip() if answer_match else response

        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.7

        # Extract citations
        citations = []
        source_refs = re.findall(r'\[Source (\d+)\]', answer)
        for ref in set(source_refs):
            idx = int(ref) - 1
            if 0 <= idx < len(documents):
                doc = documents[idx]
                citations.append(Citation(
                    doc_id=doc.id,
                    text=doc.content[:100] + "...",
                    relevance=doc.relevance_score
                ))

        return answer, citations, confidence

    def _generate_fallback(self, query: str, reason: str) -> SynthesizedResponse:
        """Generate a fallback response when no good sources found."""
        return SynthesizedResponse(
            answer=f"I don't have enough information to answer this question. {reason}",
            confidence=0.0,
            citations=[],
            sources_used=0,
            is_fallback=True,
            fallback_reason=reason
        )

    def synthesize_with_verification(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> SynthesizedResponse:
        """
        Synthesize with self-verification step.
        Generates answer, then verifies it against sources.
        """
        import time
        start = time.perf_counter()

        # First pass: generate answer
        initial = self.synthesize(query, documents)

        if initial.is_fallback:
            return initial

        # Second pass: verify
        verification_prompt = f"""Verify this answer against the sources.

Question: {query}

Answer: {initial.answer}

Sources:
{self._build_context(documents)}

Check:
1. Is everything in the answer supported by the sources?
2. Are there any hallucinations or unsupported claims?
3. What confidence level (0.0-1.0) would you assign?

Respond in JSON:
{{"verified": true/false, "confidence": 0.0-1.0, "issues": ["issue 1"] or [], "corrected_answer": "if needed"}}"""

        response = self.llm_client.complete(
            verification_prompt,
            system="You verify answers against source documents.",
            temperature=0.0,
            max_tokens=500
        )

        elapsed = (time.perf_counter() - start) * 1000

        # Parse verification
        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group(0))

                # Use corrected answer if provided
                if result.get("corrected_answer"):
                    answer = result["corrected_answer"]
                else:
                    answer = initial.answer

                return SynthesizedResponse(
                    answer=answer,
                    confidence=result.get("confidence", initial.confidence),
                    citations=initial.citations,
                    sources_used=initial.sources_used,
                    tokens_used=initial.tokens_used + response.total_tokens,
                    latency_ms=elapsed
                )
        except json.JSONDecodeError:
            pass

        # Return initial if verification failed
        initial.latency_ms = elapsed
        return initial


class StreamingSynthesizer:
    """
    Synthesizer that yields tokens as they're generated.
    (For demonstration - actual streaming would need async)
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def synthesize(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> SynthesizedResponse:
        """
        Synthesize with simulated streaming.
        In production, would use actual streaming API.
        """
        context = "\n\n".join([
            f"[{i+1}] {doc.content}"
            for i, doc in enumerate(documents)
            if doc.relevance_score >= 0.4
        ])

        prompt = f"""Answer based on these sources:

{context}

Question: {query}

Answer concisely with citations [N]:"""

        response = self.llm_client.complete(
            prompt,
            system="Answer questions using provided sources.",
            temperature=0.2,
            max_tokens=500
        )

        return SynthesizedResponse(
            answer=response.content,
            confidence=0.8,
            citations=[],
            sources_used=len(documents),
            tokens_used=response.total_tokens,
            latency_ms=response.latency_ms
        )
