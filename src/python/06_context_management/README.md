# Module 6: Context Management

## The Problem

LLMs have limited context windows. When your document exceeds this limit, you must decide what to include. The wrong choice means the model can't answer questions about information it never sees.

**Example:**
- Document size: 3,229 tokens
- Available context: 1,500 tokens (after reserving for question + response)
- Challenge: Answer questions about ANY part of the document

---

## Context Management Strategies

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW STRATEGIES                            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  FULL DOCUMENT (3,229 tokens)                                    │  │
│  │  ┌────────┬────────┬────────┬────────┬────────┬────────┐        │  │
│  │  │ Ch 1-2 │ Ch 3   │ Ch 4-5 │ Ch 6-7 │ Ch 8   │ Ch 9-10│        │  │
│  │  │ Early  │ Middle │ Tech   │ Price/ │ HR     │ Metrics│        │  │
│  │  │ History│ Growth │ Arch   │ SLA    │ Policy │ Contact│        │  │
│  │  └────────┴────────┴────────┴────────┴────────┴────────┘        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─── SLIDING WINDOW (25% accuracy) ─────────────────────────────────┐ │
│  │  Only keeps: [Ch 8] [Ch 9-10]                                     │ │
│  │  Misses: Early history, acquisitions, pricing, tech architecture │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── SUMMARIZATION (25% accuracy) ──────────────────────────────────┐ │
│  │  Compressed: [Summary of all] but loses specific numbers/names   │ │
│  │  Misses: Exact prices, phone numbers, specific dates             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── HIERARCHICAL (25% accuracy) ───────────────────────────────────┐ │
│  │  [Brief summaries] + [Detailed Ch 8-10]                          │ │
│  │  Misses: Specific details from early chapters                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─── RETRIEVAL-AUGMENTED (87.5% accuracy) ──────────────────────────┐ │
│  │  Per question: Finds [most relevant chunks] via semantic search  │ │
│  │  Dynamic: Different context for each question                    │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Actual Results (Reproducible)

Run `python demo.py --provider anthropic` to reproduce:

### Accuracy Comparison

| Strategy | Accuracy | Avg Tokens | Compression |
|----------|----------|------------|-------------|
| **Sliding Window** | 25.0% | 794 | 24.6% |
| **Summarization** | 25.0% | 974 | 30.2% |
| **Hierarchical** | 25.0% | 976 | 30.2% |
| **Retrieval-Augmented** | 87.5% | 888 | 27.5% |

### Question-by-Question Analysis

| Question Location | Sliding | Summary | Hierarchical | Retrieval |
|-------------------|---------|---------|--------------|-----------|
| Ch 1: Founding date | MISS | MISS | MISS | FOUND |
| Ch 3: Acquisition price | MISS | MISS | MISS | FOUND |
| Ch 6: Pricing tier | MISS | FOUND | MISS | FOUND |
| Ch 7: SLA uptime | MISS | FOUND | FOUND | FOUND |
| Ch 8: PTO policy | FOUND | MISS | FOUND | FOUND |
| Ch 9: Retention rate | FOUND | MISS | MISS | MISS |
| Ch 10: Phone number | MISS | MISS | MISS | FOUND |
| Ch 5: AWS percentage | MISS | MISS | MISS | FOUND |

**Key Insight:** Retrieval finds relevant content regardless of document position.

---

## Strategy Implementations

### 1. Sliding Window

Keeps only the most recent chunks that fit within the limit.

```python
# strategies.py
class SlidingWindowStrategy:
    def prepare_context(self, document, question, max_tokens=1500):
        chunks = self.chunk_document(document)
        available_tokens = max_tokens - len(question) // 4 - 500

        # Add chunks from the END until limit reached
        selected = []
        current_tokens = 0

        for chunk in reversed(chunks):
            if current_tokens + len(chunk) // 4 <= available_tokens:
                selected.insert(0, chunk)
                current_tokens += len(chunk) // 4
            else:
                break

        return "\n\n".join(selected)
```

**When to use:**
- Chat conversations (recent context matters most)
- Streaming/real-time processing
- Simple implementation needed

### 2. Summarization

Compresses the entire document into summaries.

```python
# strategies.py
class SummarizationStrategy:
    def __init__(self, summarize_fn):
        self.summarize_fn = summarize_fn

    def prepare_context(self, document, question, max_tokens=1500):
        chunks = self.chunk_document(document)
        summaries = []

        for chunk in chunks:
            # Use LLM to summarize each chunk
            summary = self.summarize_fn(chunk)
            summaries.append(summary)

        return "\n\n".join(summaries)
```

**When to use:**
- General overview questions
- When you need broad coverage
- Cost-sensitive scenarios (smaller context = cheaper)

### 3. Hierarchical

Combines summaries of early content with detailed recent content.

```python
# strategies.py
class HierarchicalStrategy:
    def __init__(self, summarize_fn, summary_ratio=0.3):
        self.summarize_fn = summarize_fn
        self.summary_ratio = summary_ratio

    def prepare_context(self, document, question, max_tokens=1500):
        chunks = self.chunk_document(document)

        # Allocate: 30% for summary, 70% for recent detail
        summary_budget = int(max_tokens * 0.3)
        detail_budget = max_tokens - summary_budget

        # Summarize first 2/3 of document
        early_chunks = chunks[:len(chunks) * 2 // 3]
        summaries = [self.summarize_fn(c) for c in early_chunks]

        # Keep recent 1/3 in full detail
        late_chunks = chunks[len(chunks) * 2 // 3:]

        return f"SUMMARY:\n{summaries}\n\nDETAIL:\n{late_chunks}"
```

**When to use:**
- Mixed question types (some general, some specific)
- Documents with important recent updates
- When order matters (e.g., conversation history)

### 4. Retrieval-Augmented (Best Accuracy)

Uses semantic search to find relevant chunks per question.

```python
# strategies.py
class RetrievalAugmentedStrategy:
    def __init__(self, embed_fn, top_k=5):
        self.embed_fn = embed_fn
        self.top_k = top_k

    def prepare_context(self, document, question, max_tokens=1500):
        chunks = self.chunk_document(document)

        # Embed question
        q_embedding = self.embed_fn(question)

        # Score all chunks by similarity
        scores = []
        for chunk in chunks:
            c_embedding = self.embed_fn(chunk)
            similarity = cosine_similarity(q_embedding, c_embedding)
            scores.append((similarity, chunk))

        # Select top-k most relevant
        scores.sort(reverse=True)
        selected = [chunk for _, chunk in scores[:self.top_k]]

        return "\n\n".join(selected)
```

**When to use:**
- Specific factual questions
- Large knowledge bases
- When you can afford embedding computation

---

## Running the Demo

```bash
cd /home/adam/LLMControl
PYTHONPATH=src/python python src/python/06_context_management/demo.py --provider anthropic
```

**Expected output:**
```
CONTEXT MANAGEMENT DEMO

Document Statistics:
  Words: 1,988
  Characters: 12,918
  Estimated Tokens: 3,229

Test Questions: 8
Max Context: 1,500 tokens

STRATEGY: SLIDING WINDOW
  Q: When was TechFlow founded?
     A: The information is not in the provided context.
     Status: [WRONG]

STRATEGY: RETRIEVAL-AUGMENTED
  Q: When was TechFlow founded?
     A: March 15, 2010, by Dr. Sarah Chen, Marcus Williams, and Jennifer Park.
     Status: [CORRECT]
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `documents.py` | Long test document + questions |
| `strategies.py` | Four context management strategies |
| `demo.py` | Comparison demo |
| `README.md` | This documentation |

---

## Key Insights

### Why Retrieval Wins

1. **Question-specific context** - Different questions get different chunks
2. **No information loss** - Original text preserved, just filtered
3. **Position-independent** - Finds content anywhere in document

### Trade-offs

| Strategy | Accuracy | LLM Calls | Latency | Complexity |
|----------|----------|-----------|---------|------------|
| Sliding Window | Low | 1 | Fast | Simple |
| Summarization | Low | N+1 | Slow | Medium |
| Hierarchical | Low | M+1 | Medium | Complex |
| Retrieval | High | 1 | Medium | Medium |

*N = number of chunks, M = summary chunks*

### When Each Strategy Fails

| Strategy | Failure Mode |
|----------|-------------|
| Sliding Window | Questions about early content |
| Summarization | Specific numbers, names, dates |
| Hierarchical | Details in early sections |
| Retrieval | Questions requiring full context |

---

## Production Considerations

### Chunking Matters

Bad chunks = bad retrieval. Consider:
- **Overlap**: Include some context from previous chunk
- **Semantic boundaries**: Split at paragraphs, not mid-sentence
- **Size**: 300-500 tokens is typical sweet spot

### Hybrid Approaches

Real systems often combine strategies:
```python
def answer(document, question):
    # Retrieval for specific facts
    relevant_chunks = retrieve(document, question, top_k=3)

    # Summary for context
    overview = summarize(document)

    # Combine
    context = f"Overview:\n{overview}\n\nDetails:\n{relevant_chunks}"
    return llm(context, question)
```

### Caching

Embed documents once, query many times:
```python
# Expensive: done once per document
embeddings = [embed(chunk) for chunk in chunks]
store.save(doc_id, embeddings)

# Cheap: done per question
retrieved = store.query(embed(question), top_k=5)
```

---

## Next Steps

After mastering context management, move to:
- **Module 7: LoRA Fine-tuning** - Customize model behavior with training
