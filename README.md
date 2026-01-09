# LLMControl

A comprehensive learning project for LLM conditioning techniques and multi-agent orchestration patterns.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LLMControl                                      │
│                                                                              │
│   PART 1: Single LLM Conditioning (Python)                                  │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│   │ Prompt  │ │Structured│ │Function │ │   RAG   │ │  Guard- │              │
│   │Engineer │ │ Outputs │ │ Calling │ │         │ │  rails  │              │
│   └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘              │
│        │           │           │           │           │                     │
│   ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴────┐              │
│   │ Context │ │  LoRA   │ │  Evals  │ │ Agentic │ │   MCP   │              │
│   │ Manage  │ │Fine-tune│ │Framework│ │   RAG   │ │  Tools  │              │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘              │
│                                                                              │
│   PART 2: Multi-Agent Orchestration (TypeScript)                            │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │ Sequential  │ │  Parallel   │ │   Router/   │ │  Iterative  │          │
│   │   Chain     │ │  Fan-out    │ │ Dispatcher  │ │ Refinement  │          │
│   └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Interactive Demo App

Run the Streamlit app for interactive demos of all techniques:

```bash
cd app
streamlit run Home.py
```

**Features:**
- 11 interactive demo pages covering all modules
- Multi-test comparison benchmark (6 test categories)
- Custom scoring calculator for your use case
- Dark mode compatible styling

---

## Quick Start

```bash
# Clone and setup
cd LLMControl

# Install Python dependencies
pip install -r requirements.txt

# Set up API key
cp config/.env.example config/.env
# Edit config/.env with your ANTHROPIC_API_KEY

# Run the interactive app
cd app && streamlit run Home.py

# Or run any module demo directly
cd src/python/01_prompt_engineering
python demo.py
```

---

## Project Structure

```
LLMControl/
├── app/                        # Streamlit interactive demo app
│   ├── Home.py                 # Main dashboard
│   ├── styles.py               # Shared styling (dark mode compatible)
│   └── pages/                  # 11 demo pages
│       ├── 01_Prompt_Engineering.py
│       ├── 02_Structured_Outputs.py
│       ├── 03_Function_Calling.py
│       ├── 04_RAG.py
│       ├── 05_Guardrails.py
│       ├── 06_Context_Management.py
│       ├── 07_LoRA_Finetuning.py
│       ├── 08_Evals.py
│       ├── 09_Agentic_RAG.py
│       ├── 10_MCP_Integration.py
│       └── 11_Comparison.py    # Multi-test benchmark
├── config/
│   └── .env                    # API keys (ANTHROPIC_API_KEY)
├── src/
│   ├── python/                 # Part 1: LLM Conditioning (10 modules)
│   │   ├── shared/             # Shared utilities
│   │   ├── 01_prompt_engineering/
│   │   ├── 02_structured_outputs/
│   │   ├── 03_function_calling/
│   │   ├── 04_rag/
│   │   ├── 05_guardrails/
│   │   ├── 06_context_management/
│   │   ├── 07_lora_finetuning/
│   │   ├── 08_evals_framework/
│   │   ├── 09_agentic_rag/
│   │   └── 10_mcp_integration/
│   └── typescript/             # Part 2: Multi-Agent Patterns (4 patterns)
│       └── src/
│           ├── shared/
│           ├── 01_sequential_chain/
│           ├── 02_parallel_fanout/
│           ├── 03_router_dispatcher/
│           └── 04_iterative_refinement/
└── README.md
```

---

## Part 1: Single LLM Conditioning (Python)

### Module 1: Prompt Engineering
**Crafting effective prompts to guide LLM behavior.**

```bash
cd src/python/01_prompt_engineering && python demo.py
```

| Technique | Accuracy | Use Case |
|-----------|----------|----------|
| Zero-shot | 75% | Simple tasks |
| Few-shot | 100% | Pattern matching |
| Chain-of-thought | 100% | Complex reasoning |

📁 [View Module](src/python/01_prompt_engineering/)

---

### Module 2: Structured Outputs
**Forcing LLMs to return predictable formats.**

```bash
cd src/python/02_structured_outputs && python demo.py
```

| Method | Parse Success | Best For |
|--------|---------------|----------|
| Raw text | 87.5% | Simple extraction |
| JSON mode | 87.5% | Structured data |
| Tool use | 100% | Complex schemas |

📁 [View Module](src/python/02_structured_outputs/)

---

### Module 3: Function Calling
**Teaching LLMs to invoke external tools.**

```bash
cd src/python/03_function_calling && python demo.py
```

| Scenario | Without Tools | With Tools |
|----------|---------------|------------|
| Calculator | 0% | 100% |
| Weather API | 0% | 100% |
| Database | 0% | 100% |

📁 [View Module](src/python/03_function_calling/)

---

### Module 4: RAG (Retrieval Augmented Generation)
**Enhancing LLM responses with external knowledge.**

```bash
cd src/python/04_rag && python demo.py
```

| Metric | Without RAG | With RAG |
|--------|-------------|----------|
| Accuracy | 0% | 100% |
| Hallucination | High | Low |

📁 [View Module](src/python/04_rag/)

---

### Module 5: Guardrails
**Safety layers for input/output validation.**

```bash
cd src/python/05_guardrails && python demo.py
```

| Check | Result |
|-------|--------|
| PII Detection | 40% blocked |
| Prompt Injection | 30% sanitized |
| Content Moderation | 30% passed |

📁 [View Module](src/python/05_guardrails/)

---

### Module 6: Context Management
**Strategies for handling long contexts efficiently.**

```bash
cd src/python/06_context_management && python demo.py
```

| Strategy | Accuracy | Token Efficiency |
|----------|----------|------------------|
| Sliding Window | 25% | High |
| Summarization | 25% | High |
| Hierarchical | 25% | Medium |
| Retrieval-Augmented | 87.5% | Best |

📁 [View Module](src/python/06_context_management/)

---

### Module 7: LoRA Fine-Tuning
**Adapting models with minimal parameters.**

```bash
cd src/python/07_lora_finetuning && python demo.py
```

| Metric | Value |
|--------|-------|
| Trainable Params | 0.2% |
| Training Time | 3.8 min |
| GPU Memory | 4GB (GTX 1650) |

📁 [View Module](src/python/07_lora_finetuning/)

---

### Module 8: Evals Framework ⭐ NEW
**Systematic evaluation of LLM outputs.**

```bash
cd src/python/08_evals_framework && python demo.py
```

| Evaluator | Cost | Speed | Use Case |
|-----------|------|-------|----------|
| Code-based | $0 | <1ms | Format validation |
| Semantic | $0 | ~10ms | Meaning comparison |
| LLM Judge | ~$0.0002 | ~1-2s | Quality assessment |

📁 [View Module](src/python/08_evals_framework/)

---

### Module 9: Agentic RAG ⭐ NEW
**Self-correcting RAG with feedback loops.**

```bash
cd src/python/09_agentic_rag && python demo.py
```

| Metric | Basic RAG | Agentic RAG |
|--------|-----------|-------------|
| Accuracy | 75% | 100% |
| Self-correction | No | Yes |
| Query Rewriting | No | Yes |

📁 [View Module](src/python/09_agentic_rag/)

---

### Module 10: MCP Integration ⭐ NEW
**Model Context Protocol for tool integration.**

```bash
cd src/python/10_mcp_integration && python demo.py
```

| Tool | Description |
|------|-------------|
| `rag_search` | Search knowledge base |
| `rag_query` | Ask questions with RAG |
| `eval_llm_judge` | LLM quality evaluation |
| `eval_faithfulness` | Hallucination detection |

📁 [View Module](src/python/10_mcp_integration/)

---

## Part 2: Multi-Agent Orchestration (TypeScript)

```bash
cd src/typescript
npm install
```

### Pattern 1: Sequential Chain
**Pipeline processing: A → B → C → Output**

```bash
npm run demo:sequential
```

```
[Researcher] → [Writer] → [Editor] → [Formatter]
     ↓            ↓           ↓            ↓
  Research     Draft       Polish      Format
```

| Agent | Tokens | Time |
|-------|--------|------|
| Researcher | 560 | 4.2s |
| Writer | 1217 | 5.4s |
| Editor | 1385 | 5.2s |
| **Total** | **4309** | **17.8s** |

📁 [View Pattern](src/typescript/src/01_sequential_chain/)

---

### Pattern 2: Parallel Fan-out
**Concurrent processing with aggregation**

```bash
npm run demo:parallel
```

```
         ┌→ [Financial] ─┐
[Query] ─┼→ [Technical] ─┼→ [Synthesizer]
         └→ [Market]    ─┘
```

| Metric | Sequential | Parallel |
|--------|------------|----------|
| Time | 14.82s | 4.89s |
| **Speedup** | 1x | **3.03x** |

📁 [View Pattern](src/typescript/src/02_parallel_fanout/)

---

### Pattern 3: Router/Dispatcher
**Intelligent routing to specialists**

```bash
npm run demo:router
```

```
              ┌→ [Technical Support]
[Classifier] ─┼→ [Billing Support]
              └→ [Sales Support]
```

| Test Case | Route | Confidence |
|-----------|-------|------------|
| API errors | technical | 90% |
| Duplicate charge | billing | 90% |
| Enterprise pricing | sales | 90% |

📁 [View Pattern](src/typescript/src/03_router_dispatcher/)

---

### Pattern 4: Iterative Refinement
**Generator-critic loop for quality**

```bash
npm run demo:iterative
```

```
[Generator] → [Critic] → [Generator] → ... → [Final]
     ↑           │
     └───────────┘ (feedback loop)
```

| Iteration | Score | Status |
|-----------|-------|--------|
| 1 | 90% | ✓ Approved |

📁 [View Pattern](src/typescript/src/04_iterative_refinement/)

---

## Key Concepts by Module

| Module | Key Concept | Improvement |
|--------|-------------|-------------|
| 01 Prompt | Chain-of-thought | 25% → 100% accuracy |
| 02 Structured | Tool use | 100% parse success |
| 03 Functions | Tool calling | Enable external APIs |
| 04 RAG | Vector retrieval | 0% → 100% accuracy |
| 05 Guardrails | Safety layers | Block harmful content |
| 06 Context | Retrieval-augmented | 87.5% with limits |
| 07 LoRA | Parameter efficiency | 0.2% trainable |
| 08 Evals | LLM-as-judge | Automated QA |
| 09 Agentic RAG | Self-correction | +25% accuracy |
| 10 MCP | Tool protocol | Claude Desktop integration |

---

## Dependencies

### Python
```
anthropic>=0.18.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
peft>=0.7.0
transformers>=4.36.0
torch>=2.0.0
mcp>=1.0.0
```

### TypeScript
```
@anthropic-ai/sdk
zod
tsx
```

---

## Claude Desktop Integration

Add to `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llmcontrol": {
      "command": "python",
      "args": ["/path/to/LLMControl/src/python/10_mcp_integration/server.py"]
    }
  }
}
```

Then use in Claude Desktop:
- "Search the knowledge base for leave policy"
- "What is the password expiration policy?"
- "Evaluate this response for hallucinations"

---

## Learning Path

### Beginner
1. **Prompt Engineering** - Foundation for everything
2. **Structured Outputs** - Reliable data extraction
3. **Function Calling** - Extend LLM capabilities

### Intermediate
4. **RAG** - Add external knowledge
5. **Guardrails** - Production safety
6. **Context Management** - Handle long documents

### Advanced
7. **LoRA Fine-tuning** - Custom model behavior
8. **Evals Framework** - Quality measurement
9. **Agentic RAG** - Self-improving systems
10. **MCP Integration** - Tool ecosystem

### Multi-Agent
11. **Sequential Chain** - Pipeline processing
12. **Parallel Fan-out** - Concurrent analysis
13. **Router/Dispatcher** - Task routing
14. **Iterative Refinement** - Quality loops

---

## Technique Comparison Benchmark

The app includes a multi-test benchmark comparing all techniques across 6 test categories:

| Category | Description | Complexity |
|----------|-------------|------------|
| Factual Accuracy | Direct Q&A with verifiable answers | Low |
| Reasoning Accuracy | Multi-step logic and inference | High |
| Format Accuracy | Output in specific JSON/structure | Medium |
| Internal Knowledge | Company docs, policies, internal data | Medium |
| Web Knowledge | Real-time web info, current events | High |
| Ambiguous Queries | Vague queries needing clarification | High |

**Key Findings:**

| Technique | Overall | Best At |
|-----------|---------|---------|
| Agentic RAG | 87.6% | Web knowledge (95%), Ambiguous queries (92%) |
| Few-shot + RAG | 83.0% | Internal knowledge (94%), Balanced |
| Basic RAG | 73.5% | Internal knowledge (92%) |
| Chain-of-thought | 53.6% | Reasoning (82%) |
| Structured Output | 54.1% | Format compliance (95%) |
| Few-shot | 52.5% | Factual accuracy (92%) |
| Zero-shot | 39.5% | Low cost ($0.0001), Fast (480ms) |

**Decision Matrix:**

| Priority | Best Choice |
|----------|-------------|
| Low cost, simple tasks | Zero-shot |
| Complex reasoning | Chain-of-thought |
| Strict JSON format | Structured Output |
| Internal docs/data | Basic RAG |
| Web/real-time data | Agentic RAG |
| Vague/ambiguous queries | Agentic RAG |

---

## Performance Summary

| Technique | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Prompt Engineering | 75% | 100% | +25% |
| Structured Outputs | 87.5% | 100% | +12.5% |
| RAG | 0% | 100% | +100% |
| Agentic RAG | 75% | 100% | +25% |
| Parallel Fan-out | 1x | 3x | 3x speedup |

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new modules
4. Submit a pull request

---

## Acknowledgments

- Built with [Anthropic Claude](https://anthropic.com)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- Fine-tuning with [PEFT](https://github.com/huggingface/peft)
- MCP SDK from [Anthropic](https://github.com/anthropics/mcp)
