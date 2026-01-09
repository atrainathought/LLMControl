#!/usr/bin/env python3
"""
Context Management Demo

This script demonstrates different strategies for handling documents
that exceed context window limits:

1. Sliding Window - Keep recent content only
2. Summarization - Compress document into summaries
3. Hierarchical - Summary + recent detail
4. Retrieval-Augmented - Semantic search for relevant chunks

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

from documents import LONG_DOCUMENT, TEST_QUESTIONS, get_document_stats
from strategies import (
    SlidingWindowStrategy,
    SummarizationStrategy,
    HierarchicalStrategy,
    RetrievalAugmentedStrategy,
    create_summarizer,
    create_embedder,
)


def answer_question(client, context: str, question: str) -> str:
    """Generate an answer using the LLM with given context."""
    prompt = f"""Based on the following document excerpt, answer the question.
If the answer is not in the provided context, say "Information not found in context."

Document:
{context}

Question: {question}

Answer:"""

    response = client.complete(prompt, temperature=0.0, max_tokens=200)
    return response.content.strip()


def check_answer(answer: str, expected_contains: list) -> bool:
    """Check if answer contains expected information."""
    answer_lower = answer.lower()
    return all(term.lower() in answer_lower for term in expected_contains)


def run_strategy_test(
    strategy_name: str,
    strategy,
    client,
    questions: list,
    verbose: bool = True
) -> dict:
    """Test a context strategy on all questions."""
    results = {
        "strategy": strategy_name,
        "correct": 0,
        "total": len(questions),
        "details": [],
        "avg_tokens": 0,
        "avg_compression": 0,
    }

    total_tokens = 0
    total_compression = 0

    for q in questions:
        if verbose:
            print(f"\n  Q: {q['question'][:60]}...")

        # Prepare context using strategy
        context_result = strategy.prepare_context(
            LONG_DOCUMENT,
            q['question'],
            max_tokens=1500  # Smaller limit to demonstrate tradeoffs
        )

        total_tokens += context_result.final_tokens
        total_compression += context_result.compression_ratio

        # Get answer
        answer = answer_question(client, context_result.context_used, q['question'])

        # Check correctness
        is_correct = check_answer(answer, q['answer_contains'])

        if is_correct:
            results["correct"] += 1

        if verbose:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"     A: {answer[:80]}...")
            print(f"     Status: [{status}] (expected: {q['answer_contains']})")
            print(f"     Tokens: {context_result.final_tokens} ({context_result.compression_ratio:.1%} of original)")

        results["details"].append({
            "question": q['question'],
            "answer": answer,
            "correct": is_correct,
            "tokens_used": context_result.final_tokens,
            "compression": context_result.compression_ratio,
            "location": q['location']
        })

    results["avg_tokens"] = total_tokens / len(questions)
    results["avg_compression"] = total_compression / len(questions)

    return results


def main():
    parser = argparse.ArgumentParser(description="Context Management Demo")
    parser.add_argument(
        "--provider",
        choices=["anthropic"],
        default="anthropic",
        help="LLM provider to use",
    )
    args = parser.parse_args()

    # Initialize
    tracker = MetricsTracker("06_context_management")

    try:
        client = AnthropicClient()
    except Exception as e:
        print(f"Error: Could not initialize Anthropic client: {e}")
        return 1

    # Document stats
    stats = get_document_stats()

    print("\n" + "=" * 70)
    print("CONTEXT MANAGEMENT DEMO")
    print("=" * 70)
    print(f"\nDocument Statistics:")
    print(f"  Words: {stats['words']:,}")
    print(f"  Characters: {stats['characters']:,}")
    print(f"  Estimated Tokens: {stats['estimated_tokens']:,}")
    print(f"\nTest Questions: {len(TEST_QUESTIONS)}")
    print(f"Max Context: 1,500 tokens (simulating limited window)")
    print("=" * 70)

    # Initialize strategies
    summarizer = create_summarizer(client)
    embedder = create_embedder()

    strategies = [
        ("Sliding Window", SlidingWindowStrategy(chunk_size=500)),
        ("Summarization", SummarizationStrategy(summarize_fn=summarizer, chunk_size=500)),
        ("Hierarchical", HierarchicalStrategy(summarize_fn=summarizer, summary_ratio=0.3, chunk_size=500)),
    ]

    # Add retrieval strategy if embeddings available
    if embedder:
        strategies.append(
            ("Retrieval-Augmented", RetrievalAugmentedStrategy(embed_fn=embedder, top_k=5, chunk_size=500))
        )
    else:
        print("\n  [Note: sentence-transformers not available, skipping retrieval strategy]")

    all_results = []

    for strategy_name, strategy in strategies:
        print("\n" + "=" * 70)
        print(f"STRATEGY: {strategy_name.upper()}")
        print("=" * 70)

        results = run_strategy_test(
            strategy_name,
            strategy,
            client,
            TEST_QUESTIONS,
            verbose=True
        )

        all_results.append(results)

        accuracy = results["correct"] / results["total"] * 100
        print(f"\n  Results: {results['correct']}/{results['total']} ({accuracy:.1f}%)")
        print(f"  Avg Tokens Used: {results['avg_tokens']:.0f}")
        print(f"  Avg Compression: {results['avg_compression']:.1%}")

        # Track metrics
        exp = tracker.create_experiment(
            experiment_name="context_strategy",
            approach=strategy_name.lower().replace("-", "_").replace(" ", "_"),
            provider="anthropic",
            model=client.model,
        )

        for detail in results["details"]:
            exp.add_sample(
                correct=detail["correct"],
                latency_ms=0,  # Not measuring per-question latency in this demo
            )

    # Summary comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Strategy':<25} {'Accuracy':<12} {'Avg Tokens':<12} {'Compression':<12}")
    print("-" * 60)

    for r in all_results:
        accuracy = r["correct"] / r["total"] * 100
        print(f"{r['strategy']:<25} {accuracy:>6.1f}%     {r['avg_tokens']:>8.0f}     {r['avg_compression']:>8.1%}")

    # Detailed analysis
    print("\n" + "=" * 70)
    print("ANALYSIS BY QUESTION LOCATION")
    print("=" * 70)

    # Group questions by their location in document
    for q in TEST_QUESTIONS:
        print(f"\n  {q['location']}:")
        print(f"    Question: {q['question'][:50]}...")
        for r in all_results:
            detail = next(d for d in r["details"] if d["question"] == q["question"])
            status = "FOUND" if detail["correct"] else "MISSED"
            print(f"    {r['strategy']:<20}: [{status}]")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
CONTEXT MANAGEMENT STRATEGIES:

1. SLIDING WINDOW (Recent Content Only)
   - Simple implementation
   - Works well for questions about recent/end content
   - Misses information from earlier sections
   - No LLM overhead for preparation

2. SUMMARIZATION (Compressed Overview)
   - Covers full document breadth
   - Good for general/overview questions
   - May lose specific details (numbers, names)
   - Requires LLM calls for summarization (adds cost/latency)

3. HIERARCHICAL (Summary + Detail)
   - Balanced approach: global context + recent specifics
   - Best for mixed question types
   - More complex implementation
   - Moderate LLM overhead

4. RETRIEVAL-AUGMENTED (Semantic Search)
   - Best accuracy for specific questions
   - Only includes relevant chunks
   - Requires embedding model
   - May miss context if question doesn't match document phrasing

RECOMMENDATIONS:
- Use RETRIEVAL for specific lookups ("what is X?")
- Use HIERARCHICAL for mixed/exploratory queries
- Use SLIDING WINDOW for conversation continuity
- Use SUMMARIZATION when you need broad coverage on a budget
""")

    print(tracker.compare())
    tracker.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
