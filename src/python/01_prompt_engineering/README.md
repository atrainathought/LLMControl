# Module 1: Prompt Engineering

## What is Prompt Engineering?

Prompt engineering is the art of crafting effective instructions for Large Language Models (LLMs) to guide their behavior **without changing the model itself**. It's the most accessible and cost-effective way to improve LLM performance.

Think of it like giving directions: the same destination can be reached with vague directions ("go that way") or precise ones ("turn left at the light, then take the second right"). The LLM's response quality depends heavily on how you ask.

---

## Four Techniques Demonstrated

### 1. Zero-Shot Prompting

**What it is:** Asking the model to perform a task directly, with no examples.

**When to use:**
- Simple, well-defined tasks
- When the model already understands the domain
- When you want minimal token usage

**Example:**
```
Classify this review as POSITIVE, NEGATIVE, or NEUTRAL:
"Amazing product! Works perfectly."

Response: POSITIVE
```

**Pros:** Lowest token cost, fastest response, simple to implement
**Cons:** May be inconsistent for ambiguous cases, no guidance on edge cases

---

### 2. Few-Shot Prompting

**What it is:** Providing examples in the prompt to teach the model the expected pattern.

**When to use:**
- Complex or nuanced tasks
- When you need consistent output format
- When zero-shot gives inconsistent results

**Example:**
```
Classify reviews as POSITIVE, NEGATIVE, or NEUTRAL.

Example 1:
Review: "Love it! Best purchase ever."
Sentiment: POSITIVE

Example 2:
Review: "Terrible quality, broke immediately."
Sentiment: NEGATIVE

Now classify:
Review: "It works okay, nothing special."
Sentiment: NEUTRAL
```

**Pros:** More consistent output format, handles edge cases better
**Cons:** Higher token cost, need to choose good examples

**Best practices:**
- Use 3-5 diverse examples
- Include edge cases in examples
- Order examples strategically

---

### 3. Chain-of-Thought (CoT) Prompting - Original

**What it is:** Asking the model to reason through the problem step-by-step.

**Original prompt (PROBLEMATIC):**
```
Think through this step-by-step:
1. Identify key positive words or phrases
2. Identify key negative words or phrases
3. Consider the overall tone and context
4. Make your final classification
```

**The Problem:** When the model lists "positive words: none" or "negative words: disappointing", our simple keyword parser finds "POSITIVE" or "NEGATIVE" in the reasoning text and returns the wrong answer.

**Result:** 50% accuracy (worse than random!)

---

### 4. Chain-of-Thought Improved - Fixed Version

**Key improvements:**
1. **Structured output marker:** Added `FINAL:` tag for reliable parsing
2. **Reframed questions:** Instead of "list positive/negative words," ask about "overall experience" and "tone"
3. **Explicit format instruction:** "You MUST end with exactly this format"

**Improved prompt:**
```
Think through this step-by-step:
1. What is the reviewer's overall experience?
2. What emotions or opinions are expressed?
3. Is the tone favorable, unfavorable, or mixed/neutral?

After your analysis, you MUST end with exactly this format:
FINAL: [your one-word classification]
```

**Result:** 100% accuracy

---

## Actual Results (Reproducible)

Run `python demo.py --provider anthropic` to reproduce:

| Approach | Accuracy | Latency | Tokens | Cost |
|----------|----------|---------|--------|------|
| **zero-shot** | 100.0% | 581ms | 824 | $0.000256 |
| **few-shot** | 100.0% | 532ms | 2,154 | $0.000618 |
| **chain-of-thought** | 50.0% | 2,017ms | 3,015 | $0.002655 |
| **chain-of-thought-improved** | 100.0% | 1,433ms | 2,420 | $0.001551 |

### Key Insights

1. **Zero-shot won on cost** - Claude Haiku already understands sentiment well, so examples weren't necessary for this simple task.

2. **Original CoT failed badly (50%)** - The parser extracted wrong labels because the model's reasoning mentioned "positive" and "negative" as analysis terms.

3. **Improved CoT recovered to 100%** - Adding a structured `FINAL:` marker and reframing the questions fixed the parsing issue.

4. **Cost scales with tokens** - Few-shot used 2.6x more tokens than zero-shot. CoT used ~3x more.

5. **Core lesson: Prompt design must account for parsing.** If you can't reliably extract the answer, accuracy means nothing.

---

## How It Works Under the Hood

LLMs are next-token predictors trained on massive text datasets. When you prompt them:

1. **Tokenization:** Your prompt is split into tokens
2. **Attention:** The model weighs which parts of the prompt are most relevant
3. **Generation:** Tokens are generated one-by-one, each influenced by the prompt

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR PROMPT                          │
│  ┌─────────────┬─────────────┬─────────────────────┐   │
│  │   System    │  Examples   │     Your Query      │   │
│  │  (context)  │  (few-shot) │                     │   │
│  └─────────────┴─────────────┴─────────────────────┘   │
│                         ↓                               │
│              Model Attention Mechanism                  │
│                         ↓                               │
│              Generated Response Tokens                  │
└─────────────────────────────────────────────────────────┘
```

**Why prompting matters:**
- The prompt establishes the "context window" that influences generation
- Examples in few-shot create a pattern the model continues
- Chain-of-thought activates reasoning pathways learned during training
- **Output structure in prompts guides parseable responses**

---

## Running the Demo

1. **Setup environment:**
```bash
cd /home/adam/LLMControl
pip install -r requirements.txt
cp config/.env.example config/.env
# Edit config/.env with your API keys
```

2. **Run the demo:**
```bash
cd src/python/01_prompt_engineering

# With Anthropic (Claude)
python demo.py --provider anthropic

# With OpenAI
python demo.py --provider openai

# Compare both
python demo.py --provider both
```

3. **Expected output:**
```
============================================================
PROMPT ENGINEERING COMPARISON
Task: Sentiment Classification (10 product reviews)
============================================================

Running ZERO-SHOT with Anthropic/claude-3-haiku-20240307
  [✓] Review 1: POSITIVE (expected: POSITIVE)
  [✓] Review 2: NEGATIVE (expected: NEGATIVE)
  ...
  Accuracy: 100.0% (10/10)
  Avg Latency: 581ms
  Total Cost: $0.000256
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `prompts.py` | All 4 prompt templates + test dataset + extraction logic |
| `demo.py` | Main demo script that runs all experiments |
| `README.md` | This documentation |

---

## The Prompt Engineering Iteration Loop

This module demonstrates the core prompt engineering workflow:

```
┌──────────────────────────────────────────────────────────┐
│  1. Write initial prompt                                 │
│              ↓                                           │
│  2. Run on test data, measure accuracy                   │
│              ↓                                           │
│  3. Analyze failures (why did it get X wrong?)           │
│              ↓                                           │
│  4. Identify root cause (parsing? reasoning? ambiguity?) │
│              ↓                                           │
│  5. Modify prompt to address root cause                  │
│              ↓                                           │
│  6. Re-run and compare (did accuracy improve?)           │
│              ↓                                           │
│  7. Repeat until satisfied                               │
└──────────────────────────────────────────────────────────┘
```

In this module:
- **Original CoT:** 50% accuracy (parsing failure)
- **Root cause:** Model output contained "positive/negative" as analysis terms
- **Fix:** Structured `FINAL:` marker + reframed questions
- **Improved CoT:** 100% accuracy

---

## Key Takeaways

1. **Start with zero-shot** - It's your free baseline
2. **Add few-shot examples** when consistency matters
3. **Use chain-of-thought** for complex reasoning tasks
4. **Design prompts for parsing** - Unstructured output = unreliable extraction
5. **Measure everything** - accuracy, latency, cost
6. **Iterate empirically** - prompt engineering is experimental science

---

## Next Steps

After mastering prompt engineering, move to:
- **Module 2: Structured Outputs** - Force reliable output formats with schemas
- **Module 4: RAG** - Add external knowledge to prompts
