# Module 8: Evals Framework

Systematic evaluation of LLM outputs using automated methods.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EVALS FRAMEWORK                               │
│                                                                      │
│   Test Cases    ┌─────────────────┐                                  │
│   ──────────→   │  LLM Response   │                                  │
│                 └────────┬────────┘                                  │
│                          │                                           │
│            ┌─────────────┼─────────────┐                            │
│            ▼             ▼             ▼                            │
│     ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│     │Code-Based│  │ Semantic │  │LLM Judge │                       │
│     │Evaluator │  │Evaluator │  │Evaluator │                       │
│     └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
│          │             │             │                              │
│          └─────────────┴─────────────┘                              │
│                        │                                             │
│                        ▼                                             │
│              ┌─────────────────┐                                    │
│              │   EvalReport    │                                    │
│              │  (Pass/Fail,    │                                    │
│              │   Scores,       │                                    │
│              │   Metrics)      │                                    │
│              └─────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
cd src/python/08_evals_framework
python demo.py
```

---

## Three Types of Evaluators

### 1. Code-Based Evaluators (Deterministic)

Fast, reliable checks that don't require ML models.

| Evaluator | Use Case | Example |
|-----------|----------|---------|
| `ExactMatchEvaluator` | Exact answer matching | "Paris" == "Paris" |
| `ContainsAllEvaluator` | Keyword presence | Contains ["Python", "AI"] |
| `RegexMatchEvaluator` | Pattern matching | Phone: `\d{3}-\d{3}-\d{4}` |
| `JSONSchemaEvaluator` | JSON structure validation | Has {name: str, age: int} |
| `LengthEvaluator` | Response length bounds | 50-500 characters |

```python
from evaluators import ExactMatchEvaluator, ContainsAllEvaluator

# Exact match (case insensitive)
evaluator = ExactMatchEvaluator(case_sensitive=False)
result = evaluator.evaluate("Paris", "paris")
print(result.passed)  # True

# Keyword checking
evaluator = ContainsAllEvaluator()
result = evaluator.evaluate(
    "Python is great for machine learning",
    ["Python", "machine learning"]
)
print(result.score)  # 1.0
```

### 2. Semantic Evaluator (Embedding-Based)

Compares meaning using sentence embeddings.

```python
from evaluators import SemanticSimilarityEvaluator

evaluator = SemanticSimilarityEvaluator(threshold=0.7)

result = evaluator.evaluate(
    "The cat sat on the mat.",
    "A feline was resting on a rug."
)
print(result.score)  # ~0.85 (high similarity)
```

**When to use:**
- Paraphrase detection
- Answer variations (same meaning, different words)
- Semantic search evaluation

### 3. LLM-as-Judge Evaluator

Uses Claude to evaluate quality, relevance, and faithfulness.

```python
from evaluators import LLMJudgeEvaluator, FaithfulnessEvaluator
from shared.llm_client import AnthropicClient

client = AnthropicClient()

# General quality assessment
judge = LLMJudgeEvaluator(client)
result = judge.evaluate(
    actual="Regular exercise improves cardiovascular health...",
    question="What are the benefits of exercise?"
)
print(result.score)      # 0.85
print(result.reasoning)  # "The response is accurate and comprehensive..."

# RAG faithfulness check
faithfulness = FaithfulnessEvaluator(client)
result = faithfulness.evaluate(
    actual="Revenue was $150 million in 2024.",
    context="Annual Report: Revenue reached $150M..."
)
print(result.passed)  # True (grounded in context)
```

---

## Evaluation Pipeline

Run test suites systematically with multiple evaluators.

```python
from pipeline import EvalPipeline, TestCase
from evaluators import ContainsAllEvaluator, SemanticSimilarityEvaluator

# Create test cases
test_cases = [
    TestCase(
        id="q1",
        input="What is the capital of Japan?",
        expected="Tokyo"
    ),
    TestCase(
        id="q2",
        input="Who wrote Romeo and Juliet?",
        expected="William Shakespeare"
    ),
]

# Set up pipeline
pipeline = EvalPipeline(
    llm_client=client,
    system_prompt="Answer concisely.",
    temperature=0.0,
)
pipeline.add_evaluator(ContainsAllEvaluator())
pipeline.add_evaluator(SemanticSimilarityEvaluator(threshold=0.6))

# Run evaluation
report = pipeline.run(test_cases)
report.print_summary()
```

**Output:**
```
============================================================
Evaluation Pipeline Report
============================================================

ACCURACY:
  Accuracy: 100.00%
  Pass Rate: 100.00%

QUALITY:
  Average Score: 92.50%
  Consistency: 95.00%

EFFICIENCY:
  Avg Latency: 1234.00
  Total Cost: 0.00
============================================================
```

---

## Pre-Built Test Suites

```python
from test_cases import get_test_suite

# Available suites
qa_tests = get_test_suite("qa")              # Factual Q&A
rag_tests = get_test_suite("rag")            # RAG with context
classification_tests = get_test_suite("classification")  # Sentiment, routing
code_tests = get_test_suite("code")          # Code generation
```

### Create Custom Test Suite

```python
from test_cases import create_custom_test_suite

my_tests = create_custom_test_suite(
    questions=["What is Python?", "What is JavaScript?"],
    expected_answers=["A programming language", "A web scripting language"],
    prefix="custom"
)
```

---

## Composite Evaluator

Combine multiple evaluators with weighted scoring.

```python
from evaluators import CompositeEvaluator, LengthEvaluator, SemanticSimilarityEvaluator

composite = CompositeEvaluator()
composite.add_evaluator(LengthEvaluator(min_length=50), weight=0.2)
composite.add_evaluator(SemanticSimilarityEvaluator(), weight=0.8)

result = composite.evaluate(actual, expected)
print(result.score)  # Weighted average
```

---

## Model Comparison

Compare multiple models on the same test suite.

```python
from pipeline import ModelComparison

comparison = ModelComparison(test_cases, evaluators)
comparison.add_model("claude-haiku", haiku_client)
comparison.add_model("claude-sonnet", sonnet_client)
comparison.print_comparison()
```

**Output:**
```
================================================================
MODEL COMPARISON
================================================================
Metric                   claude-haiku        claude-sonnet
----------------------------------------------------------------
Accuracy                 92.00%              98.00%
Average Score            88.50%              95.20%
Avg Latency              856.00              2341.00
================================================================
```

---

## Metrics Available

### Accuracy Metrics
- `accuracy(correct, total)` - Simple accuracy
- `precision(tp, fp)` - True positive rate
- `recall(tp, fn)` - Sensitivity
- `f1_score(p, r)` - Harmonic mean

### Quality Metrics
- `average_score(scores)` - Mean score
- `pass_rate(passed, total)` - % above threshold
- `consistency(scores)` - Low variance = high consistency

### Retrieval Metrics (RAG)
- `retrieval_precision()` - Relevant/Retrieved
- `retrieval_recall()` - Relevant retrieved/Total relevant
- `mean_reciprocal_rank()` - MRR score
- `ndcg_at_k()` - Normalized DCG
- `faithfulness_score()` - Groundedness

### Efficiency Metrics
- `average_latency()` - Response time
- `total_cost()` - API costs
- `cost_per_eval()` - Cost efficiency

---

## Files

```
08_evals_framework/
├── __init__.py        # Package exports
├── evaluators.py      # All evaluator classes
├── metrics.py         # Metric calculations
├── pipeline.py        # EvalPipeline orchestration
├── test_cases.py      # Sample test suites
├── demo.py            # Comprehensive demo
└── README.md          # This file
```

---

## Demo Results

```
======================================================================
CODE-BASED EVALUATORS
======================================================================
1. ExactMatchEvaluator
  Exact match: 'Paris' vs 'Paris' → True (score: 1.0)
  Case insensitive: 'paris' vs 'Paris' → True (score: 1.0)

2. ContainsAllEvaluator
  Found 3/3 keywords → True (score: 1.0)

3. JSONSchemaEvaluator
  Valid JSON: True (score: 1.0)
  Invalid JSON: False (score: 0.5)
  Errors: ["Field 'age' should be int, got str"]

======================================================================
SEMANTIC SIMILARITY EVALUATOR
======================================================================
  "The cat sat on the mat." vs "A feline was resting on a rug."
  Score: 85.23% → ✓ PASS

  "Python is great for AI." vs "Python excels at machine learning."
  Score: 78.45% → ✓ PASS

======================================================================
LLM-AS-JUDGE EVALUATOR
======================================================================
  Judge Score: 88.00%
  Reasoning: "The response is accurate, well-structured..."
  Strengths: Comprehensive, Accurate, Well-organized
  Judge Cost: $0.000234

======================================================================
FAITHFULNESS EVALUATOR (RAG)
======================================================================
  Faithful Response: Score: 95.00% → ✓ FAITHFUL
  Hallucinated Response: Score: 35.00% → ✗ HALLUCINATION
    Hallucinations: ["founded in 2005", "2000 employees", "IPO"]
```

---

## Cost Analysis

| Evaluator Type | Cost per Eval | Speed | Best For |
|----------------|---------------|-------|----------|
| Code-Based | $0 | <1ms | Format validation |
| Semantic | $0 (local) | ~10ms | Meaning comparison |
| LLM Judge | ~$0.0002 | ~1-2s | Quality assessment |

**Recommendation:** Use code-based for high-volume checks, LLM judge for sampled quality assessment.

---

## Use Cases

### 1. CI/CD Quality Gates
```python
# Fail build if accuracy drops below 90%
report = pipeline.run(regression_tests)
if report.metrics.get_summary()["accuracy"].value < 0.9:
    raise Exception("Quality regression detected!")
```

### 2. RAG Evaluation
```python
# Check faithfulness of RAG responses
pipeline.add_evaluator(FaithfulnessEvaluator(client))
report = pipeline.run(rag_test_cases)
hallucination_rate = 1 - report.metrics.get_summary()["pass_rate"].value
```

### 3. A/B Testing Models
```python
comparison = ModelComparison(tests, evaluators)
comparison.add_model("baseline", baseline_client)
comparison.add_model("candidate", candidate_client)
comparison.print_comparison()
```

### 4. Prompt Optimization
```python
# Compare different prompts
for prompt in prompts:
    pipeline.system_prompt = prompt
    report = pipeline.run(tests)
    print(f"{prompt[:30]}: {report.metrics.get_summary()['accuracy'].value:.2%}")
```

---

## Best Practices

1. **Layer evaluators**: Code-based first (fast/cheap), then semantic, then LLM judge
2. **Sample for LLM judge**: Don't run on every output, sample 10-20%
3. **Version test suites**: Track test cases alongside code
4. **Monitor over time**: Store metrics for trend analysis
5. **Human correlation**: Periodically check LLM judge agrees with humans

---

## Next Steps

This module provides the foundation for:
- **Module 9: Agentic RAG** - Uses evals for self-correction
- **Module 10: MCP Integration** - Exposes evals as MCP tools
