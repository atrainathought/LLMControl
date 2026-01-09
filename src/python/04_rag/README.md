# Module 4: RAG (Retrieval Augmented Generation)

## What is RAG?

RAG (Retrieval Augmented Generation) enhances LLM responses by retrieving relevant external knowledge before generating an answer. Instead of relying solely on training data, the model gets access to specific, up-to-date documents.

**The core problem:** LLMs don't know your company's policies, your product documentation, or any information created after their training cutoff. RAG solves this by injecting relevant context into the prompt.

---

## How RAG Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE                                  │
│                                                                         │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────────┐        │
│  │  User    │───>│   Generate       │───>│   Search Vector    │        │
│  │  Query   │    │   Query Embedding │    │   Store (Top-K)    │        │
│  └──────────┘    └──────────────────┘    └─────────┬──────────┘        │
│                                                    │                    │
│                                                    ↓                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    Retrieved Chunks                          │      │
│  │   [Source 1: leave_policy]  "20 days annual leave..."       │      │
│  │   [Source 2: leave_policy]  "Parental leave: 16 weeks..."   │      │
│  │   [Source 3: expenses]      "Daily meal limit: $75..."      │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                 │                                       │
│                                 ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │               Context-Augmented Prompt                        │      │
│  │                                                               │      │
│  │   "Use the following context to answer the question..."      │      │
│  │                                                               │      │
│  │   Context:                                                    │      │
│  │   [Source 1] "20 days annual leave..."                       │      │
│  │   [Source 2] "Parental leave: 16 weeks..."                   │      │
│  │                                                               │      │
│  │   Question: "How many days of annual leave do employees get?" │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                 │                                       │
│                                 ↓                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                      LLM Generation                           │      │
│  │                                                               │      │
│  │   "According to the TechFlow Employee Handbook, all          │      │
│  │    full-time employees are entitled to 20 days of paid       │      │
│  │    annual leave per year."                                    │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Actual Results (Reproducible)

Run `python demo.py --provider anthropic` to reproduce:

| Approach | Accuracy | Avg Latency | Questions Answered |
|----------|----------|-------------|-------------------|
| **With RAG** | 100% | 866ms | 8/8 correct |
| **Without RAG** | 0% | 1,406ms | 0/8 correct |

### Sample Results

| Question | RAG Answer | Without RAG |
|----------|-----------|-------------|
| "How many days annual leave?" | "20 days per year" ✓ | "I don't have access to TechFlow's policies" ✗ |
| "Password expiration policy?" | "90 days" ✓ | "I don't have access to internal policies" ✗ |
| "Meal expense limit domestic?" | "$75 per day" ✓ | "I don't know the specific limit" ✗ |
| "DataSync Pro uptime SLA?" | "99.99%" ✓ | "I don't have enough information" ✗ |

---

## RAG Pipeline Components

### 1. Document Chunking

Split large documents into smaller pieces for precise retrieval.

```python
# documents.py
def chunk_by_paragraphs(doc: Document, max_chunk_size: int = 500) -> List[Chunk]:
    """Split document into chunks by paragraphs."""
    paragraphs = doc.content.split('\n\n')
    # Combine small paragraphs, split large ones
    ...
```

**Why chunk?**
- Embeddings work best on shorter text (512 tokens typical)
- Retrieves only relevant sections, not entire documents
- Reduces noise in the context

**Chunking strategies:**
- By paragraphs (simple, works well)
- By headers/sections (preserves document structure)
- Fixed size with overlap (ensures no content lost)

### 2. Embedding Generation

Convert text to dense vectors that capture semantic meaning.

```python
# vectorstore.py
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()
```

**Key concepts:**
- Text → 384-dimensional vector (for MiniLM)
- Similar texts have similar vectors
- Enables semantic search (not just keyword matching)

### 3. Vector Storage (ChromaDB)

Store embeddings for fast similarity search.

```python
# vectorstore.py
import chromadb

class VectorStore:
    def __init__(self, collection_name="rag_demo"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

    def search(self, query: str, n_results: int = 3):
        query_embedding = self.embedder.embed([query])[0]
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
```

**Why ChromaDB?**
- Simple Python API
- No external server needed
- Supports persistence
- Fast similarity search (HNSW algorithm)

### 4. Retrieval + Generation

Combine retrieved context with the question.

```python
# pipeline.py
def query(self, question: str) -> RAGResult:
    # 1. Retrieve similar chunks
    chunks = self.vector_store.search(question, n_results=3)

    # 2. Build context
    context = "\n\n".join(chunk["content"] for chunk in chunks)

    # 3. Generate with context
    prompt = f"""Use the following context to answer the question.

    Context:
    {context}

    Question: {question}

    Answer:"""

    response = self.llm_client.complete(prompt)
    return response
```

---

## Key Insights

### Why RAG Achieved 100% Accuracy

1. **Relevant Context Retrieved:** Semantic search found the right chunks for each question
2. **Grounded Answers:** LLM could quote directly from source documents
3. **No Hallucination:** Model acknowledged when info came from specific sources

### Why Without RAG Failed (0%)

1. **No Access to Private Data:** LLM's training data doesn't include TechFlow's policies
2. **Honest Acknowledgment:** Claude correctly said "I don't have access" rather than guessing
3. **No Hallucination:** Model didn't make up fake policies

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge updates** | Just add new documents | Requires retraining |
| **Source attribution** | Can cite specific documents | Knowledge is "baked in" |
| **Cost** | Only embedding cost upfront | Training costs |
| **Latency** | Retrieval adds ~20ms | No extra latency |
| **Best for** | Factual Q&A, documentation | Style/format changes |

---

## Running the Demo

```bash
# Install dependencies
pip install chromadb sentence-transformers

# Run demo
cd /home/adam/LLMControl
PYTHONPATH=src/python python src/python/04_rag/demo.py --provider anthropic
```

**Expected output:**
```
======================================================================
RAG (RETRIEVAL AUGMENTED GENERATION) DEMO
======================================================================

STEP 1: Document Chunking
  Documents: 5
  Chunks created: 17

STEP 2: Create Vector Store
  Using SentenceTransformer embeddings (all-MiniLM-L6-v2)
  Vector store size: 17 chunks

EXPERIMENT: WITH RAG vs WITHOUT RAG

Question 1: How many days of annual leave do employees get at TechFlow?
  WITH RAG:
    Retrieved 3 chunks: leave_policy (0.769), leave_policy (0.513), oncall (0.475)
    Answer: "20 days of paid annual leave per year"
    Correct: ✓

  WITHOUT RAG:
    Answer: "I don't have access to TechFlow's internal documentation"
    Correct: ✗
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `documents.py` | Sample knowledge base + chunking strategies |
| `vectorstore.py` | Embedding generation + ChromaDB storage |
| `pipeline.py` | RAG query pipeline + baseline comparison |
| `demo.py` | Full comparison demo |
| `README.md` | This documentation |

---

## Knowledge Base Used

The demo uses fictional "TechFlow Inc." documentation:

| Document | Topic | Sample Questions |
|----------|-------|------------------|
| Employee Handbook | Leave policies | "How many days annual leave?" |
| IT Security Policy | Password, data | "Password expiration policy?" |
| Expense Policy | Travel, meals | "Daily meal limit domestic?" |
| Product Docs | DataSync Pro | "Uptime SLA?" |
| Engineering Policy | On-call | "SEV1 response time?" |

---

## Common RAG Improvements

1. **Hybrid Search:** Combine semantic + keyword search
2. **Reranking:** Use a cross-encoder to rerank initial results
3. **Query Expansion:** Expand query with synonyms/related terms
4. **Chunk Overlap:** Ensure context isn't lost at chunk boundaries
5. **Metadata Filtering:** Filter by category before semantic search

---

## Next Steps

After mastering RAG, move to:
- **Module 5: Guardrails** - Add safety layers to RAG outputs
- **Module 6: Context Management** - Handle long contexts efficiently
