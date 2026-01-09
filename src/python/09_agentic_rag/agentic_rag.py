"""
Agentic RAG Orchestrator.

Self-correcting RAG system that:
1. Retrieves documents
2. Evaluates relevance
3. Rewrites query if needed
4. Re-retrieves with better query
5. Synthesizes final response

This creates a feedback loop that improves answer quality.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import Retriever, RetrievalResult, RetrievedDocument
from evaluator import RelevanceEvaluator, RelevanceEvaluation, QuickRelevanceEvaluator
from rewriter import QueryRewriter, RewriteResult
from synthesizer import ResponseSynthesizer, SynthesizedResponse


class AgentAction(Enum):
    """Actions the agent can take."""
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    REWRITE = "rewrite"
    SYNTHESIZE = "synthesize"
    FALLBACK = "fallback"


@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    action: AgentAction
    query: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticRAGResult:
    """Complete result from agentic RAG."""
    answer: str
    confidence: float
    iterations: int
    final_query: str
    steps: List[AgentStep]
    total_docs_retrieved: int
    relevant_docs_used: int
    total_tokens: int
    total_latency_ms: float
    success: bool

    def print_trace(self):
        """Print execution trace."""
        print(f"\n{'='*60}")
        print("AGENTIC RAG EXECUTION TRACE")
        print(f"{'='*60}")
        print(f"Original Query: {self.steps[0].query if self.steps else 'N/A'}")
        print(f"Final Query: {self.final_query}")
        print(f"Iterations: {self.iterations}")
        print(f"Success: {self.success}")
        print(f"\n--- Steps ---")

        for i, step in enumerate(self.steps, 1):
            print(f"\n{i}. {step.action.value.upper()}")
            print(f"   Query: {step.query[:50]}...")
            for key, value in step.details.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2%}")
                elif isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                else:
                    print(f"   {key}: {value}")

        print(f"\n--- Summary ---")
        print(f"Docs Retrieved: {self.total_docs_retrieved}")
        print(f"Relevant Docs Used: {self.relevant_docs_used}")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"Total Latency: {self.total_latency_ms:.0f}ms")
        print(f"Confidence: {self.confidence:.0%}")
        print(f"{'='*60}")


class AgenticRAG:
    """
    Self-correcting RAG system.

    Loop:
    1. Retrieve documents
    2. Evaluate relevance
    3. If insufficient → rewrite query and retry
    4. If sufficient → synthesize response
    """

    def __init__(
        self,
        retriever: Retriever,
        evaluator: RelevanceEvaluator,
        rewriter: QueryRewriter,
        synthesizer: ResponseSynthesizer,
        max_iterations: int = 3,
        use_quick_eval: bool = False
    ):
        self.retriever = retriever
        self.evaluator = evaluator
        self.rewriter = rewriter
        self.synthesizer = synthesizer
        self.max_iterations = max_iterations
        self.use_quick_eval = use_quick_eval

        if use_quick_eval:
            self.quick_evaluator = QuickRelevanceEvaluator()

    def query(self, question: str, verbose: bool = False) -> AgenticRAGResult:
        """
        Execute agentic RAG query.

        Args:
            question: The user's question
            verbose: Print progress

        Returns:
            AgenticRAGResult with answer and execution trace
        """
        steps = []
        current_query = question
        total_tokens = 0
        total_latency = 0
        all_docs = []
        iteration = 0

        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n--- Iteration {iteration} ---")
                print(f"Query: {current_query}")

            # Step 1: Retrieve
            retrieval = self.retriever.retrieve(current_query)
            all_docs.extend(retrieval.documents)
            total_latency += retrieval.retrieval_time_ms

            steps.append(AgentStep(
                action=AgentAction.RETRIEVE,
                query=current_query,
                details={
                    "docs_found": len(retrieval.documents),
                    "avg_similarity": retrieval.avg_similarity,
                    "strategy": retrieval.strategy
                }
            ))

            if verbose:
                print(f"Retrieved {len(retrieval.documents)} docs (avg sim: {retrieval.avg_similarity:.2%})")

            # Step 2: Evaluate
            if self.use_quick_eval:
                evaluation = self.quick_evaluator.evaluate(current_query, retrieval)
            else:
                evaluation = self.evaluator.evaluate(current_query, retrieval)

            steps.append(AgentStep(
                action=AgentAction.EVALUATE,
                query=current_query,
                details={
                    "overall_relevance": evaluation.overall_relevance,
                    "has_sufficient": evaluation.has_sufficient_context,
                    "recommendation": evaluation.recommendation
                }
            ))

            if verbose:
                print(f"Relevance: {evaluation.overall_relevance:.2%}, Recommendation: {evaluation.recommendation}")

            # Step 3: Decide next action
            if evaluation.recommendation == "proceed":
                # Good enough - synthesize
                if verbose:
                    print("Proceeding to synthesis...")

                relevant_docs = [d for d in retrieval.documents if d.relevance_score >= 0.4]
                response = self.synthesizer.synthesize(question, relevant_docs)

                total_tokens += response.tokens_used
                total_latency += response.latency_ms

                steps.append(AgentStep(
                    action=AgentAction.SYNTHESIZE,
                    query=current_query,
                    details={
                        "confidence": response.confidence,
                        "sources_used": response.sources_used,
                        "is_fallback": response.is_fallback
                    }
                ))

                return AgenticRAGResult(
                    answer=response.answer,
                    confidence=response.confidence,
                    iterations=iteration,
                    final_query=current_query,
                    steps=steps,
                    total_docs_retrieved=len(all_docs),
                    relevant_docs_used=response.sources_used,
                    total_tokens=total_tokens,
                    total_latency_ms=total_latency,
                    success=not response.is_fallback
                )

            elif evaluation.recommendation in ["rewrite", "expand"]:
                # Need to rewrite query
                if verbose:
                    print(f"Rewriting query ({evaluation.recommendation})...")

                rewrite = self.rewriter.rewrite(
                    current_query,
                    failed_docs=[d.content[:200] for d in retrieval.documents],
                    strategy=evaluation.recommendation if evaluation.recommendation == "expand" else "auto"
                )

                steps.append(AgentStep(
                    action=AgentAction.REWRITE,
                    query=current_query,
                    details={
                        "strategy": rewrite.strategy,
                        "new_query": rewrite.rewritten_query[:100],
                        "reasoning": rewrite.reasoning
                    }
                ))

                current_query = rewrite.rewritten_query

                if verbose:
                    print(f"New query: {current_query}")

            else:
                # Fallback - can't find relevant info
                if verbose:
                    print("No relevant information found, falling back...")
                break

        # Max iterations or fallback
        steps.append(AgentStep(
            action=AgentAction.FALLBACK,
            query=current_query,
            details={"reason": "Max iterations reached or no relevant docs"}
        ))

        return AgenticRAGResult(
            answer=f"I couldn't find sufficient information to answer: {question}",
            confidence=0.0,
            iterations=iteration,
            final_query=current_query,
            steps=steps,
            total_docs_retrieved=len(all_docs),
            relevant_docs_used=0,
            total_tokens=total_tokens,
            total_latency_ms=total_latency,
            success=False
        )


class BasicRAG:
    """
    Simple RAG without self-correction (for comparison).
    """

    def __init__(self, retriever: Retriever, synthesizer: ResponseSynthesizer):
        self.retriever = retriever
        self.synthesizer = synthesizer

    def query(self, question: str) -> AgenticRAGResult:
        """Simple retrieve-and-generate without feedback loop."""
        # Retrieve
        retrieval = self.retriever.retrieve(question)

        # Synthesize directly (no evaluation/rewriting)
        response = self.synthesizer.synthesize(question, retrieval.documents)

        return AgenticRAGResult(
            answer=response.answer,
            confidence=response.confidence,
            iterations=1,
            final_query=question,
            steps=[
                AgentStep(AgentAction.RETRIEVE, question, {"docs": len(retrieval.documents)}),
                AgentStep(AgentAction.SYNTHESIZE, question, {"confidence": response.confidence})
            ],
            total_docs_retrieved=len(retrieval.documents),
            relevant_docs_used=response.sources_used,
            total_tokens=response.tokens_used,
            total_latency_ms=retrieval.retrieval_time_ms + response.latency_ms,
            success=not response.is_fallback
        )


def create_agentic_rag(vectorstore, llm_client, max_iterations: int = 3) -> AgenticRAG:
    """
    Factory function to create a fully configured AgenticRAG.
    """
    retriever = Retriever(vectorstore, n_results=3)
    evaluator = RelevanceEvaluator(llm_client, threshold=0.5)
    rewriter = QueryRewriter(llm_client)
    synthesizer = ResponseSynthesizer(llm_client)

    return AgenticRAG(
        retriever=retriever,
        evaluator=evaluator,
        rewriter=rewriter,
        synthesizer=synthesizer,
        max_iterations=max_iterations
    )


def create_basic_rag(vectorstore, llm_client) -> BasicRAG:
    """Factory function to create basic RAG for comparison."""
    retriever = Retriever(vectorstore, n_results=3)
    synthesizer = ResponseSynthesizer(llm_client)

    return BasicRAG(retriever, synthesizer)
