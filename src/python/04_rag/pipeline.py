"""
RAG Pipeline

This module implements the full RAG workflow:
1. Retrieve relevant chunks from vector store
2. Build context-augmented prompt
3. Generate answer using LLM
4. Compare with baseline (no RAG)
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from vectorstore import VectorStore


@dataclass
class RAGResult:
    """Result of a RAG query."""
    question: str
    answer: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    context_used: str = ""
    latency_ms: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_latency_ms: float = 0
    generation_latency_ms: float = 0


class RAGPipeline:
    """
    Complete RAG pipeline for question answering.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client,  # AnthropicClient
        n_retrieve: int = 3,
        verbose: bool = True
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.n_retrieve = n_retrieve
        self.verbose = verbose

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk['metadata'].get('topic', chunk['metadata'].get('doc_id', 'unknown'))
            context_parts.append(f"[Source {i+1}: {source}]\n{chunk['content']}")
        return "\n\n---\n\n".join(context_parts)

    def _build_rag_prompt(self, question: str, context: str) -> str:
        """Build the RAG prompt with context."""
        return f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer based on the context above:"""

    def query(self, question: str) -> RAGResult:
        """
        Run full RAG pipeline:
        1. Retrieve relevant chunks
        2. Build context-augmented prompt
        3. Generate answer
        """
        result = RAGResult(question=question)

        # Step 1: Retrieve
        retrieval_start = time.perf_counter()
        chunks = self.vector_store.search(question, n_results=self.n_retrieve)
        result.retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
        result.retrieved_chunks = chunks

        if self.verbose:
            print(f"  Retrieved {len(chunks)} chunks:")
            for chunk in chunks:
                print(f"    - {chunk['metadata'].get('topic', 'unknown')} (similarity: {chunk['similarity']:.3f})")

        # Step 2: Build context
        result.context_used = self._build_context(chunks)

        # Step 3: Generate
        prompt = self._build_rag_prompt(question, result.context_used)

        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Be concise and specific. Quote relevant information when appropriate.
If the context doesn't contain the answer, say so clearly."""

        generation_start = time.perf_counter()
        response = self.llm_client.complete(
            prompt,
            system=system_prompt,
            temperature=0.0
        )
        result.generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        result.answer = response.content
        result.input_tokens = response.input_tokens
        result.output_tokens = response.output_tokens
        result.latency_ms = result.retrieval_latency_ms + result.generation_latency_ms

        return result


def query_without_rag(
    llm_client,
    question: str,
    verbose: bool = True
) -> RAGResult:
    """
    Query without RAG for comparison.

    The LLM must rely on its training data alone.
    """
    result = RAGResult(question=question)

    system_prompt = """You are a helpful assistant. Answer the question based on your knowledge.
If you don't know the answer, say so clearly.
Note: You do NOT have access to TechFlow Inc.'s internal documentation."""

    prompt = f"Question: {question}\n\nAnswer:"

    start_time = time.perf_counter()
    response = llm_client.complete(
        prompt,
        system=system_prompt,
        temperature=0.0
    )
    result.latency_ms = (time.perf_counter() - start_time) * 1000
    result.generation_latency_ms = result.latency_ms

    result.answer = response.content
    result.input_tokens = response.input_tokens
    result.output_tokens = response.output_tokens

    return result


def evaluate_answer(
    predicted: str,
    expected: str,
    question: str
) -> Dict[str, Any]:
    """
    Evaluate if the predicted answer contains the expected information.

    Simple evaluation: check if key parts of expected answer appear in prediction.
    """
    predicted_lower = predicted.lower()
    expected_lower = expected.lower()

    # Extract key terms from expected answer
    # Remove common words and check for key terms
    key_terms = [term.strip() for term in expected_lower.split() if len(term) > 2]

    # Check if main numeric or specific values appear
    import re
    numbers = re.findall(r'\d+(?:\.\d+)?', expected)
    key_values = re.findall(r'\$\d+|\d+%|\d+ (?:days?|weeks?|hours?|minutes?)', expected_lower)

    # Score based on key value presence
    matches = 0
    total = 0

    for num in numbers:
        total += 1
        if num in predicted:
            matches += 1

    for val in key_values:
        total += 1
        if val in predicted_lower:
            matches += 1

    if total == 0:
        # Fall back to checking if expected text is substring
        correct = expected_lower in predicted_lower or \
                  any(term in predicted_lower for term in expected_lower.split()[:3])
    else:
        correct = matches >= total * 0.5  # At least half of key values present

    return {
        "correct": correct,
        "expected": expected,
        "predicted_snippet": predicted[:200] + "..." if len(predicted) > 200 else predicted,
        "key_values_found": matches,
        "key_values_total": total
    }
