#!/usr/bin/env python3
"""
Evals Framework Demo.

Demonstrates all three types of evaluators:
1. Code-based evaluators (exact match, contains, regex, JSON schema)
2. Semantic similarity evaluators (embedding-based)
3. LLM-as-judge evaluators (using Claude to evaluate)

Also shows the evaluation pipeline for running test suites.
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_client import AnthropicClient

from evaluators import (
    ExactMatchEvaluator,
    ContainsAllEvaluator,
    RegexMatchEvaluator,
    JSONSchemaEvaluator,
    LengthEvaluator,
    SemanticSimilarityEvaluator,
    LLMJudgeEvaluator,
    FaithfulnessEvaluator,
    CompositeEvaluator,
)
from pipeline import EvalPipeline, TestCase
from test_cases import get_test_suite, RAG_TEST_CASES
from metrics import EvalMetrics


def demo_code_based_evaluators():
    """Demonstrate code-based evaluators."""
    print("=" * 70)
    print("CODE-BASED EVALUATORS")
    print("=" * 70)
    print("\nDeterministic checks that don't require ML models.\n")

    # 1. Exact Match
    print("-" * 40)
    print("1. ExactMatchEvaluator")
    print("-" * 40)
    exact_eval = ExactMatchEvaluator(case_sensitive=False)

    test_pairs = [
        ("Paris", "Paris", "Exact match"),
        ("paris", "Paris", "Case insensitive"),
        ("New York", "Paris", "No match"),
    ]

    for actual, expected, desc in test_pairs:
        result = exact_eval.evaluate(actual, expected)
        print(f"  {desc}: '{actual}' vs '{expected}' → {result.passed} (score: {result.score})")

    # 2. Contains All
    print("\n" + "-" * 40)
    print("2. ContainsAllEvaluator")
    print("-" * 40)
    contains_eval = ContainsAllEvaluator()

    response = "Python is a high-level programming language known for its simplicity and readability."
    keywords = ["Python", "programming", "language"]
    result = contains_eval.evaluate(response, keywords)
    print(f"  Response: '{response[:50]}...'")
    print(f"  Keywords: {keywords}")
    print(f"  Result: {result.passed} (score: {result.score})")
    print(f"  Details: {result.details}")

    # 3. Regex Match
    print("\n" + "-" * 40)
    print("3. RegexMatchEvaluator")
    print("-" * 40)
    regex_eval = RegexMatchEvaluator()

    patterns = [
        (r"\d{3}-\d{3}-\d{4}", "My phone is 555-123-4567", "Phone number"),
        (r"[A-Z]{2,4}\d{3,6}", "Order ID: ABC12345", "Order ID"),
        (r"^[A-Z]", "hello world", "Starts with capital"),
    ]

    for pattern, text, desc in patterns:
        result = regex_eval.evaluate(text, pattern)
        print(f"  {desc}: '{pattern}' in '{text[:30]}' → {result.passed}")

    # 4. JSON Schema
    print("\n" + "-" * 40)
    print("4. JSONSchemaEvaluator")
    print("-" * 40)
    json_eval = JSONSchemaEvaluator()

    schema = {"name": str, "age": int, "active": bool}
    valid_json = '{"name": "Alice", "age": 30, "active": true}'
    invalid_json = '{"name": "Bob", "age": "thirty"}'

    result = json_eval.evaluate(valid_json, schema)
    print(f"  Valid JSON: {result.passed} (score: {result.score})")

    result = json_eval.evaluate(invalid_json, schema)
    print(f"  Invalid JSON: {result.passed} (score: {result.score})")
    print(f"  Errors: {result.details.get('errors', [])}")

    # 5. Length
    print("\n" + "-" * 40)
    print("5. LengthEvaluator")
    print("-" * 40)
    length_eval = LengthEvaluator(min_length=10, max_length=100)

    texts = [
        "Short",
        "This is a reasonable length response that fits the criteria.",
        "X" * 150,
    ]

    for text in texts:
        result = length_eval.evaluate(text)
        print(f"  Length {len(text)}: {result.passed} (score: {result.score:.2f})")


def demo_semantic_evaluator():
    """Demonstrate semantic similarity evaluator."""
    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY EVALUATOR")
    print("=" * 70)
    print("\nUses sentence embeddings to compare meaning.\n")

    semantic_eval = SemanticSimilarityEvaluator(threshold=0.7)

    pairs = [
        ("The cat sat on the mat.", "A feline was resting on a rug.", "Similar meaning"),
        ("Python is great for AI.", "Python excels at machine learning.", "Related topic"),
        ("The weather is sunny.", "I love pizza.", "Unrelated"),
        ("Capital of France is Paris.", "Paris is France's capital city.", "Same fact, different phrasing"),
    ]

    for actual, expected, desc in pairs:
        result = semantic_eval.evaluate(actual, expected)
        print(f"\n  {desc}:")
        print(f"    Actual:   '{actual}'")
        print(f"    Expected: '{expected}'")
        print(f"    Score: {result.score:.2%} → {'✓ PASS' if result.passed else '✗ FAIL'}")


def demo_llm_judge(client):
    """Demonstrate LLM-as-judge evaluator."""
    print("\n" + "=" * 70)
    print("LLM-AS-JUDGE EVALUATOR")
    print("=" * 70)
    print("\nUses Claude to evaluate response quality.\n")

    judge = LLMJudgeEvaluator(client)

    # Test case: Evaluate a response
    question = "What are the benefits of regular exercise?"
    response = """Regular exercise offers numerous health benefits:
    1. Improved cardiovascular health and reduced risk of heart disease
    2. Better weight management and metabolism
    3. Enhanced mood and reduced symptoms of depression
    4. Stronger bones and muscles
    5. Better sleep quality
    6. Increased energy levels throughout the day"""

    print("-" * 40)
    print("Evaluating response quality")
    print("-" * 40)
    print(f"Question: {question}")
    print(f"Response: {response[:100]}...")

    result = judge.evaluate(response, question=question)
    print(f"\nJudge Score: {result.score:.2%}")
    print(f"Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}")
    if result.details.get("strengths"):
        print(f"Strengths: {', '.join(result.details['strengths'][:3])}")
    if result.details.get("issues"):
        print(f"Issues: {', '.join(result.details['issues'][:3])}")
    print(f"Judge Cost: ${result.details.get('judge_cost', 0):.6f}")


def demo_faithfulness(client):
    """Demonstrate faithfulness evaluator for RAG."""
    print("\n" + "=" * 70)
    print("FAITHFULNESS EVALUATOR (RAG)")
    print("=" * 70)
    print("\nChecks if response is grounded in context.\n")

    faithfulness = FaithfulnessEvaluator(client)

    context = """
    TechCorp Annual Report 2024:
    - Revenue: $150 million (up 25% from 2023)
    - Employees: 500 people across 5 offices
    - Founded: 2010 in San Francisco
    - Main products: Cloud storage, Data analytics, AI tools
    """

    # Faithful response
    print("-" * 40)
    print("Test 1: Faithful Response")
    print("-" * 40)
    faithful_response = "TechCorp had revenue of $150 million in 2024, a 25% increase from the previous year."
    result = faithfulness.evaluate(faithful_response, context=context)
    print(f"Response: {faithful_response}")
    print(f"Score: {result.score:.2%} → {'✓ FAITHFUL' if result.passed else '✗ HALLUCINATION'}")
    print(f"Reasoning: {result.reasoning[:200]}...")

    # Hallucinated response
    print("\n" + "-" * 40)
    print("Test 2: Hallucinated Response")
    print("-" * 40)
    hallucinated = "TechCorp was founded in 2005 and has 2000 employees. They are planning an IPO next year."
    result = faithfulness.evaluate(hallucinated, context=context)
    print(f"Response: {hallucinated}")
    print(f"Score: {result.score:.2%} → {'✓ FAITHFUL' if result.passed else '✗ HALLUCINATION'}")
    print(f"Hallucinations found: {result.details.get('hallucinations', [])}")


def demo_composite_evaluator(client):
    """Demonstrate combining multiple evaluators."""
    print("\n" + "=" * 70)
    print("COMPOSITE EVALUATOR")
    print("=" * 70)
    print("\nCombines multiple evaluators with weighted scoring.\n")

    composite = CompositeEvaluator()
    composite.add_evaluator(LengthEvaluator(min_length=50, max_length=500), weight=0.2)
    composite.add_evaluator(ContainsAllEvaluator(), weight=0.3)
    composite.add_evaluator(SemanticSimilarityEvaluator(threshold=0.6), weight=0.5)

    actual = """Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without being explicitly programmed. It uses
    algorithms and statistical models to identify patterns."""

    expected = ["machine learning", "artificial intelligence", "data", "algorithms"]

    result = composite.evaluate(actual, expected)
    print(f"Response: {actual[:80]}...")
    print(f"\nComposite Score: {result.score:.2%}")
    print(f"Passed: {result.passed}")
    print(f"\nSub-evaluations:")
    for sub in result.details.get("sub_evaluations", []):
        print(f"  {sub['evaluator']}: {sub['score']:.2%} (weight: {sub['weight']})")


def demo_eval_pipeline(client):
    """Demonstrate the full evaluation pipeline."""
    print("\n" + "=" * 70)
    print("EVALUATION PIPELINE")
    print("=" * 70)
    print("\nRuns test suites through LLM and evaluates outputs.\n")

    # Create test cases
    test_cases = [
        TestCase(
            id="factual_001",
            input="What is the capital of Japan? Answer in one word.",
            expected="Tokyo",
        ),
        TestCase(
            id="factual_002",
            input="What year did the first iPhone launch? Answer with just the year.",
            expected="2007",
        ),
        TestCase(
            id="factual_003",
            input="Who wrote '1984'? Answer with just the author's name.",
            expected="George Orwell",
        ),
        TestCase(
            id="factual_004",
            input="What is H2O commonly known as? Answer in one word.",
            expected="Water",
        ),
    ]

    # Set up pipeline
    pipeline = EvalPipeline(
        llm_client=client,
        system_prompt="Answer questions concisely. Give short, direct answers.",
        temperature=0.0,
    )
    pipeline.add_evaluator(ContainsAllEvaluator())
    pipeline.add_evaluator(SemanticSimilarityEvaluator(threshold=0.5))

    print("Running evaluation pipeline...")
    print(f"Test cases: {len(test_cases)}")
    print(f"Evaluators: {[e.name for e in pipeline.evaluators]}")
    print()

    # Run pipeline
    report = pipeline.run(test_cases)

    # Show results
    print("-" * 40)
    print("RESULTS")
    print("-" * 40)

    for run in report.runs:
        status = "✓" if run.passed else "✗"
        print(f"\n{status} {run.test_case.id}")
        print(f"  Input: {run.test_case.input[:50]}...")
        print(f"  Expected: {run.test_case.expected}")
        print(f"  Actual: {run.actual_output[:50]}...")
        print(f"  Scores: {[f'{r.evaluator_name}={r.score:.2%}' for r in run.results]}")

    # Print summary
    report.print_summary()


def demo_rag_evaluation(client):
    """Demonstrate evaluating RAG responses."""
    print("\n" + "=" * 70)
    print("RAG EVALUATION DEMO")
    print("=" * 70)
    print("\nEvaluates RAG system outputs for relevance and faithfulness.\n")

    # Simulate RAG responses (pre-computed for demo)
    rag_outputs = {
        "rag_001": "Our company offers a 30-day money-back guarantee for unused items in their original packaging.",
        "rag_002": "TechCorp employs approximately 500 people across 5 offices worldwide.",
        "rag_003": "The minimum requirements are Windows 10 or later, 8GB RAM, and 500MB of disk space.",
        "rag_004": "Our API has official SDKs for Python, JavaScript, Java, and Go.",
    }

    # Set up pipeline with RAG-specific evaluators
    pipeline = EvalPipeline()
    pipeline.add_evaluator(SemanticSimilarityEvaluator(threshold=0.6))
    pipeline.add_evaluator(FaithfulnessEvaluator(client))

    print("Evaluating RAG responses...")
    print(f"Test cases: {len(RAG_TEST_CASES)}")
    print()

    # Run with pre-computed outputs
    report = pipeline.run(RAG_TEST_CASES, pre_computed_outputs=rag_outputs)

    # Show results
    print("-" * 40)
    print("RAG EVALUATION RESULTS")
    print("-" * 40)

    for run in report.runs:
        status = "✓" if run.passed else "✗"
        print(f"\n{status} {run.test_case.id}: {run.test_case.input[:40]}...")

        for result in run.results:
            print(f"  {result.evaluator_name}: {result.score:.2%}")
            if result.eval_type.value == "llm_judge":
                if result.details.get("hallucinations"):
                    print(f"    Hallucinations: {result.details['hallucinations']}")

    report.print_summary()


def main():
    """Run all demos."""
    print("=" * 70)
    print("EVALS FRAMEWORK DEMO")
    print("=" * 70)
    print("""
This module provides three types of evaluators for LLM outputs:

1. CODE-BASED: Deterministic checks (exact match, regex, JSON, length)
2. SEMANTIC: Embedding-based similarity comparison
3. LLM-AS-JUDGE: Uses an LLM to evaluate quality and faithfulness

Plus an EvalPipeline for running test suites systematically.
""")

    # Demo code-based evaluators (no API needed)
    demo_code_based_evaluators()

    # Demo semantic evaluator (needs sentence-transformers)
    try:
        demo_semantic_evaluator()
    except ImportError as e:
        print(f"\nSkipping semantic demo: {e}")
        print("Install with: pip install sentence-transformers")

    # Initialize LLM client for LLM-based evaluators
    print("\n" + "=" * 70)
    print("Initializing LLM client for judge-based evaluators...")
    print("=" * 70)

    try:
        client = AnthropicClient()
        print(f"Using: {client.get_name()}")

        # Run LLM-based demos
        demo_llm_judge(client)
        demo_faithfulness(client)
        demo_composite_evaluator(client)
        demo_eval_pipeline(client)
        demo_rag_evaluation(client)

    except Exception as e:
        print(f"\nError with LLM client: {e}")
        print("Skipping LLM-based demos. Check your API key.")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
KEY INSIGHTS:

1. CODE-BASED EVALUATORS
   - Fast and deterministic
   - Best for: structured outputs, format validation
   - Cost: $0 (no API calls)

2. SEMANTIC EVALUATORS
   - Good for meaning comparison
   - Best for: paraphrase detection, similar answers
   - Cost: Local model only (sentence-transformers)

3. LLM-AS-JUDGE EVALUATORS
   - Most flexible and nuanced
   - Best for: quality assessment, faithfulness checking
   - Cost: ~$0.0001-0.001 per evaluation (varies by model)

4. COMPOSITE EVALUATORS
   - Combine multiple methods
   - Weighted scoring for balanced assessment
   - Use when single metric is insufficient

5. EVAL PIPELINE
   - Systematically evaluate test suites
   - Track metrics across evaluations
   - Compare models and detect regressions

NEXT STEPS:
- Create custom test suites for your use case
- Use evals in CI/CD to catch regressions
- Track metrics over time for improvement
""")


if __name__ == "__main__":
    main()
