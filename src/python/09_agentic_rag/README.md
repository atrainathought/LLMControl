# Module 9: Agentic RAG

Self-correcting RAG with feedback loops that dynamically evaluates and re-retrieves when needed.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENTIC RAG LOOP                            │
│                                                                      │
│    ┌─────────┐    ┌─────────────┐    ┌──────────────┐               │
│    │  Query  │───▶│  Retrieve   │───▶│  Evaluate    │               │
│    └─────────┘    │  Documents  │    │  Relevance   │               │
│         ▲         └─────────────┘    └──────┬───────┘               │
│         │                                    │                       │
│         │         ┌──────────────────────────┼──────────────┐       │
│         │         │                          ▼              │       │
│         │         │    ┌─────────────────────────────┐      │       │
│         │         │    │      Relevance Score        │      │       │
│         │         │    │    ┌───────────────────┐    │      │       │
│         │         │    │    │  < threshold?     │    │      │       │
│         │         │    │    └─────────┬─────────┘    │      │       │
│         │         │    └──────────────┼──────────────┘      │       │
│         │         │                   │                     │       │
│         │         │     YES           │          NO         │       │
│         │         │     ▼             │          ▼          │       │
│    ┌────┴────┐    │  ┌─────────┐      │    ┌──────────┐     │       │
│    │ Rewrite │◀───┼──│ Rewrite │      │    │Synthesize│     │       │
│    │  Query  │    │  │  Query  │      │    │ Response │     │       │
│    └─────────┘    │  └─────────┘      │    └────┬─────┘     │       │
│                   │                   │         │           │       │
│                   └───────────────────┼─────────┼───────────┘       │
│                                       │         │                    │
│                                       │         ▼                    │
│                                       │   ┌──────────┐               │
│                                       │   │  Answer  │               │
│                                       │   │  + Cite  │               │
│                                       │   └──────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
cd src/python/09_agentic_rag
python demo.py
```

---

## Key Concepts

### Basic RAG vs Agentic RAG

| Aspect | Basic RAG | Agentic RAG |
|--------|-----------|-------------|
| Flow | Retrieve → Generate | Retrieve → Evaluate → [Rewrite] → Generate |
| Self-correction | No | Yes |
| Query rewriting | No | Yes |
| Confidence scoring | Basic | LLM-evaluated |
| Iterations | 1 | 1-3+ |

### The Self-Correction Loop

```python
for iteration in range(max_iterations):
    # 1. Retrieve documents
    docs = retriever.retrieve(query)

    # 2. Evaluate relevance (LLM judges each doc)
    evaluation = evaluator.evaluate(query, docs)

    if evaluation.has_sufficient_context:
        # 3a. Good enough - synthesize answer
        return synthesizer.generate(query, docs)
    else:
        # 3b. Not good enough - rewrite query
        query = rewriter.rewrite(query, failed_docs=docs)
        # Loop continues with new query
```

---

## Components

### 1. Retriever (`retriever.py`)

Enhanced retrieval with multiple strategies.

```python
from retriever import Retriever

retriever = Retriever(vectorstore, n_results=3)

# Standard semantic search
result = retriever.retrieve("annual leave policy")

# Hybrid search (semantic + keyword boost)
result = retriever.retrieve_hybrid("password expiration", keywords=["90", "days"])

# Query expansion (search with multiple related queries)
result = retriever.retrieve_with_expansion(
    "PTO policy",
    expanded_queries=["annual leave", "vacation days", "time off"]
)
```

### 2. Evaluator (`evaluator.py`)

LLM-based relevance grading.

```python
from evaluator import RelevanceEvaluator

evaluator = RelevanceEvaluator(llm_client, threshold=0.6)

evaluation = evaluator.evaluate(query, retrieval_result)

print(evaluation.overall_relevance)  # 0.75
print(evaluation.has_sufficient_context)  # True
print(evaluation.recommendation)  # "proceed" | "rewrite" | "expand" | "fallback"

# Each document is graded
for grade in evaluation.grades:
    print(f"{grade.doc_id}: {grade.label} ({grade.score:.0%})")
    # doc_001: relevant (85%)
    # doc_002: partial (60%)
    # doc_003: irrelevant (20%)
```

### 3. Rewriter (`rewriter.py`)

Query transformation strategies.

```python
from rewriter import QueryRewriter

rewriter = QueryRewriter(llm_client)

# Reformulate (use different terms)
result = rewriter.rewrite("PTO policy", strategy="reformulate")
# → "What is the annual leave and vacation policy?"

# Expand (add context to short queries)
result = rewriter.rewrite("password rules", strategy="expand")
# → "What are the password requirements including length, characters, and expiration?"

# Decompose (break complex questions)
result = rewriter.rewrite(
    "What's the expense limit for meals and what approval is needed?",
    strategy="decompose"
)
# → ["What is the expense limit for meals?", "What approval is needed for expenses?"]
```

### 4. Synthesizer (`synthesizer.py`)

Response generation with citations.

```python
from synthesizer import ResponseSynthesizer

synthesizer = ResponseSynthesizer(llm_client, include_citations=True)

response = synthesizer.synthesize(query, relevant_docs)

print(response.answer)
# "Employees receive 20 days of annual leave [Source 1]. Leave accrues at 1.67 days per month [Source 1]."

print(response.confidence)  # 0.85
print(response.citations)   # [Citation(doc_id="doc_001", ...)]
```

### 5. AgenticRAG (`agentic_rag.py`)

Main orchestrator combining all components.

```python
from agentic_rag import create_agentic_rag

agentic = create_agentic_rag(vectorstore, llm_client, max_iterations=3)

result = agentic.query("What's the PTO policy?", verbose=True)

print(result.answer)
print(result.confidence)
print(result.iterations)  # How many retrieval attempts
result.print_trace()  # Full execution trace
```

---

## Usage Example

```python
from shared.llm_client import AnthropicClient
from 04_rag.vectorstore import VectorStore
from 04_rag.documents import KNOWLEDGE_BASE, chunk_all_documents
from agentic_rag import create_agentic_rag

# Setup
client = AnthropicClient()
chunks = chunk_all_documents(KNOWLEDGE_BASE)
vectorstore = VectorStore()
vectorstore.add_chunks(chunks)

# Create agentic RAG
agentic = create_agentic_rag(vectorstore, client, max_iterations=3)

# Query
result = agentic.query("How do I request time off?")

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence:.0%}")
print(f"Iterations: {result.iterations}")
print(f"Success: {result.success}")

# See the full trace
result.print_trace()
```

---

## Demo Results

```
======================================================================
ACCURACY TEST (vs Ground Truth)
======================================================================

Testing 8 questions with known answers...

Q: How many days of annual leave do employees get?
   Expected: 20 days per year
   Basic: ✓ | Agentic: ✓ (iterations: 1)

Q: What is the password expiration policy?
   Expected: passwords expire every 90 days
   Basic: ✓ | Agentic: ✓ (iterations: 1)

Q: What happens if I work on a holiday?
   Expected: compensatory time off
   Basic: ✗ | Agentic: ✓ (iterations: 2)

======================================================================
ACCURACY RESULTS
======================================================================
Basic RAG:   6/8 = 75%
Agentic RAG: 8/8 = 100%
Improvement: +25%
```

---

## Query Rewriting in Action

```
======================================================================
ORIGINAL: What's the PTO policy?
======================================================================

--- Iteration 1 ---
Query: What's the PTO policy?
Retrieved 3 docs (avg sim: 45%)
Relevance: 35%, Recommendation: rewrite
Rewriting query (reformulate)...

--- Iteration 2 ---
Query: What is the annual leave and paid time off policy?
Retrieved 3 docs (avg sim: 78%)
Relevance: 85%, Recommendation: proceed
Proceeding to synthesis...

Final Answer: Employees receive 20 days of paid annual leave per year...
Confidence: 90%
Iterations: 2
```

---

## Files

```
09_agentic_rag/
├── __init__.py        # Package exports
├── retriever.py       # Enhanced retriever with scoring
├── evaluator.py       # LLM relevance grading
├── rewriter.py        # Query transformation
├── synthesizer.py     # Response generation with citations
├── agentic_rag.py     # Main orchestrator
├── demo.py            # Comparison demo
└── README.md          # This file
```

---

## Performance Metrics

| Metric | Basic RAG | Agentic RAG |
|--------|-----------|-------------|
| Accuracy | 75% | 100% |
| Avg Confidence | 70% | 85% |
| Avg Tokens | 500 | 1200 |
| Avg Latency | 1.5s | 4s |
| Self-Correction Rate | - | 25% of queries |

---

## When to Use Agentic RAG

### ✓ Good Use Cases

- **Variable query quality**: Users phrase questions differently
- **Complex knowledge base**: Technical docs, policies, procedures
- **High accuracy requirements**: Customer support, compliance
- **Ambiguous queries**: Short or vague questions

### ✗ Not Ideal For

- **Speed-critical apps**: Adds latency from multiple iterations
- **Cost-sensitive**: More LLM calls per query
- **Simple lookups**: Basic RAG is sufficient
- **Standardized queries**: Template-based questions

---

## Cost Analysis

| Component | Tokens Per Query | Cost |
|-----------|------------------|------|
| Retrieval | 0 | $0 |
| Evaluation (3 docs) | ~600 | $0.0006 |
| Rewriting (if needed) | ~200 | $0.0002 |
| Synthesis | ~400 | $0.0004 |
| **Total (1 iteration)** | ~1200 | ~$0.0012 |
| **Total (2 iterations)** | ~2000 | ~$0.0020 |

---

## Advanced Features

### HyDE (Hypothetical Document Embeddings)

```python
from rewriter import HypotheticalDocumentRewriter

hyde = HypotheticalDocumentRewriter(llm_client)
result = hyde.rewrite("password requirements")
# Generates a hypothetical answer, then searches with that
```

### Quick Evaluation (No LLM)

```python
from evaluator import QuickRelevanceEvaluator

quick_eval = QuickRelevanceEvaluator(threshold=0.5)
# Uses embedding similarity only - faster but less accurate
```

### Response Verification

```python
response = synthesizer.synthesize_with_verification(query, docs)
# Two-pass: generate, then verify against sources
```

---

## Integration with Module 8 (Evals)

```python
from 08_evals_framework import EvalPipeline, FaithfulnessEvaluator

# Evaluate agentic RAG responses for hallucinations
pipeline = EvalPipeline()
pipeline.add_evaluator(FaithfulnessEvaluator(client))

# Test the agentic RAG system
for question in test_questions:
    result = agentic.query(question)
    eval_result = pipeline.evaluators[0].evaluate(
        result.answer,
        context="\n".join([d.content for d in result.relevant_docs])
    )
    print(f"Faithfulness: {eval_result.score:.0%}")
```

---

## Next Steps

This module will be exposed as an MCP tool in **Module 10: MCP Integration**.
