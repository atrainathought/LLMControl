"""
Agentic RAG Module.

Self-correcting RAG with feedback loops:
- Relevance evaluation of retrieved documents
- Query rewriting when retrieval fails
- Iterative improvement until answer quality threshold met

Usage:
    from 09_agentic_rag import create_agentic_rag

    agentic = create_agentic_rag(vectorstore, llm_client)
    result = agentic.query("What is the leave policy?")
    print(result.answer)
    result.print_trace()
"""

from .retriever import Retriever, RetrievedDocument, RetrievalResult
from .evaluator import RelevanceEvaluator, RelevanceEvaluation, QuickRelevanceEvaluator
from .rewriter import QueryRewriter, RewriteResult, HypotheticalDocumentRewriter
from .synthesizer import ResponseSynthesizer, SynthesizedResponse
from .agentic_rag import (
    AgenticRAG,
    BasicRAG,
    AgenticRAGResult,
    AgentStep,
    AgentAction,
    create_agentic_rag,
    create_basic_rag,
)

__all__ = [
    # Retriever
    "Retriever",
    "RetrievedDocument",
    "RetrievalResult",
    # Evaluator
    "RelevanceEvaluator",
    "RelevanceEvaluation",
    "QuickRelevanceEvaluator",
    # Rewriter
    "QueryRewriter",
    "RewriteResult",
    "HypotheticalDocumentRewriter",
    # Synthesizer
    "ResponseSynthesizer",
    "SynthesizedResponse",
    # Orchestrator
    "AgenticRAG",
    "BasicRAG",
    "AgenticRAGResult",
    "AgentStep",
    "AgentAction",
    "create_agentic_rag",
    "create_basic_rag",
]
