#!/usr/bin/env python3
"""
RAG (Retrieval Augmented Generation) Demo

This script demonstrates:
1. Document chunking and embedding
2. Vector store creation with ChromaDB
3. RAG pipeline (retrieve + generate)
4. Comparison: with RAG vs without RAG

Usage:
    python demo.py --provider anthropic
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_client import AnthropicClient
from shared.metrics import MetricsTracker

from documents import KNOWLEDGE_BASE, TEST_QUESTIONS, chunk_all_documents
from vectorstore import create_vectorstore_from_documents
from pipeline import RAGPipeline, query_without_rag, evaluate_answer


def main():
    parser = argparse.ArgumentParser(description="RAG Demo")
    parser.add_argument(
        "--provider",
        choices=["anthropic"],
        default="anthropic",
        help="LLM provider to use",
    )
    args = parser.parse_args()

    # Initialize
    tracker = MetricsTracker("04_rag")

    try:
        client = AnthropicClient()
    except Exception as e:
        print(f"Error: Could not initialize Anthropic client: {e}")
        return 1

    print("\n" + "="*70)
    print("RAG (RETRIEVAL AUGMENTED GENERATION) DEMO")
    print("="*70)

    # Step 1: Chunk documents
    print("\n" + "-"*70)
    print("STEP 1: Document Chunking")
    print("-"*70)

    chunks = chunk_all_documents(KNOWLEDGE_BASE, strategy="paragraphs")
    print(f"  Documents: {len(KNOWLEDGE_BASE)}")
    print(f"  Chunks created: {len(chunks)}")
    print(f"  Sample chunk: {chunks[0].content[:100]}...")

    # Step 2: Create vector store
    print("\n" + "-"*70)
    print("STEP 2: Create Vector Store")
    print("-"*70)

    try:
        vector_store = create_vectorstore_from_documents(chunks)
        print(f"  Vector store size: {vector_store.count()} chunks")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Make sure chromadb and sentence-transformers are installed:")
        print("  pip install chromadb sentence-transformers")
        return 1

    # Step 3: Create RAG pipeline
    print("\n" + "-"*70)
    print("STEP 3: RAG Pipeline")
    print("-"*70)

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=client,
        n_retrieve=3,
        verbose=True
    )

    # Step 4: Run experiments
    print("\n" + "="*70)
    print("EXPERIMENT: WITH RAG vs WITHOUT RAG")
    print("="*70)

    # Create experiment trackers
    with_rag_exp = tracker.create_experiment(
        experiment_name="rag_comparison",
        approach="with_rag",
        provider="anthropic",
        model=client.model,
    )

    without_rag_exp = tracker.create_experiment(
        experiment_name="rag_comparison",
        approach="without_rag",
        provider="anthropic",
        model=client.model,
    )

    # Run through test questions
    for i, test in enumerate(TEST_QUESTIONS):
        question = test["question"]
        expected = test["answer"]

        print(f"\n{'='*70}")
        print(f"Question {i+1}: {question}")
        print(f"Expected: {expected}")
        print("-"*70)

        # With RAG
        print("\n  WITH RAG:")
        rag_result = rag_pipeline.query(question)
        rag_eval = evaluate_answer(rag_result.answer, expected, question)

        print(f"    Answer: {rag_result.answer[:150]}...")
        print(f"    Correct: {'✓' if rag_eval['correct'] else '✗'}")
        print(f"    Latency: {rag_result.latency_ms:.0f}ms (retrieve: {rag_result.retrieval_latency_ms:.0f}ms, generate: {rag_result.generation_latency_ms:.0f}ms)")

        with_rag_exp.add_sample(
            correct=rag_eval["correct"],
            latency_ms=rag_result.latency_ms,
            input_tokens=rag_result.input_tokens,
            output_tokens=rag_result.output_tokens,
            cost_usd=0,
        )

        # Without RAG
        print("\n  WITHOUT RAG:")
        baseline_result = query_without_rag(client, question, verbose=True)
        baseline_eval = evaluate_answer(baseline_result.answer, expected, question)

        print(f"    Answer: {baseline_result.answer[:150]}...")
        print(f"    Correct: {'✓' if baseline_eval['correct'] else '✗'}")
        print(f"    Latency: {baseline_result.latency_ms:.0f}ms")

        without_rag_exp.add_sample(
            correct=baseline_eval["correct"],
            latency_ms=baseline_result.latency_ms,
            input_tokens=baseline_result.input_tokens,
            output_tokens=baseline_result.output_tokens,
            cost_usd=0,
        )

    # Print comparison
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(tracker.compare(baseline_approach="without_rag"))

    # Save results
    tracker.save()

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print(f"""
RAG PIPELINE COMPONENTS:

1. CHUNKING
   - Split documents into smaller pieces (~500 chars)
   - Preserves semantic coherence
   - Enables precise retrieval

2. EMBEDDING
   - Convert text to dense vectors
   - Capture semantic meaning
   - Enable similarity search

3. RETRIEVAL
   - Find most similar chunks to query
   - Top-k selection (we used k=3)
   - Fast vector similarity search

4. GENERATION
   - Augment prompt with retrieved context
   - LLM generates answer from context
   - Grounded in source documents

WHY RAG WINS:
- LLM alone doesn't know TechFlow's specific policies
- RAG retrieves relevant company documentation
- Answer is grounded in actual source material
- Reduces hallucination risk
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
