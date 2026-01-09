# Module 2: Structured Outputs

## What is Structured Output?

Structured output is the practice of forcing LLMs to return data in predictable, machine-readable formats. Instead of free-form text that requires complex parsing, you get JSON, typed objects, or schema-validated responses.

**The core problem:** LLMs are trained to generate human-readable text, not machine-readable data. Without structure, you're constantly fighting parsing issues.

---

## Three Approaches Compared

### 1. Raw Text Parsing (Fragile)

**What it is:** Ask the model for analysis, then extract data with regex/string manipulation.

**Example prompt:**
```
Analyze this review and provide:
1. Sentiment (positive, negative, neutral, or mixed)
2. Confidence (0.0 to 1.0)
3. Key points
...
```

**Parsing code:**
```python
sentiment_match = re.search(r'sentiment[:\s]*(positive|negative|neutral|mixed)', response.lower())
```

**Problems:**
- Model may format output differently each time
- Regex patterns are brittle
- Edge cases break easily
- No validation of data types

**When to use:** Quick prototypes, simple extractions

---

### 2. JSON Mode (Better)

**What it is:** Ask the model to return JSON, parse with `json.loads()`.

**Example prompt:**
```
Return a JSON object with these fields:
- sentiment: "positive", "negative", "neutral", or "mixed"
- confidence: number from 0.0 to 1.0
- key_points: array of strings
...

Return ONLY valid JSON, no other text.
```

**Parsing code:**
```python
data = json.loads(response)
validated = ReviewAnalysis(**data)  # Pydantic validation
```

**Problems:**
- Model may include markdown code blocks (```json)
- Model may add explanatory text before/after JSON
- Field names may vary ("sentiment" vs "Sentiment" vs "review_sentiment")
- No guarantee of correct data types

**When to use:** Most applications, good balance of reliability and simplicity

---

### 3. Tool Use / Function Calling (Best)

**What it is:** Define a schema, and the API returns structured tool calls.

**Example schema:**
```python
{
    "name": "analyze_review",
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral", "mixed"]
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            }
        },
        "required": ["sentiment", "confidence"]
    }
}
```

**API call:**
```python
response = client.messages.create(
    model="claude-3-haiku-20240307",
    tools=[tool_schema],
    tool_choice={"type": "tool", "name": "analyze_review"},
    messages=[{"role": "user", "content": prompt}],
)

# Response contains structured tool_use block
for block in response.content:
    if block.type == "tool_use":
        data = block.input  # Already parsed and validated!
```

**Advantages:**
- Schema enforced by the API
- Correct data types guaranteed
- Enum values constrained
- No parsing required
- Built-in validation

**When to use:** Production applications, complex schemas, high reliability requirements

---

## Actual Results (Reproducible)

Run `python demo.py --provider anthropic` to reproduce:

| Approach | Parse Success | Latency | Tokens | Cost |
|----------|--------------|---------|--------|------|
| **raw_text** | 87.5% | 1,442ms | 1,922 | $0.001497 |
| **json_mode** | 87.5% | 1,147ms | 2,137 | $0.001365 |
| **tool_use** | 100.0% | 1,886ms | 7,629 | $0.003237 |

### Key Insights

1. **Tool use achieved 100% parse success** - The schema constraint prevented invalid responses.

2. **Raw text and JSON failed on ambiguous cases** - Both returned "neutral" when the correct answer was "mixed" because the prompt didn't clearly define "mixed" as an option.

3. **JSON mode was fastest and cheapest** - Less overhead than tool_use, more reliable than raw text.

4. **Tool use costs more** - The schema definition adds tokens to every request.

5. **Tradeoff: Reliability vs Cost** - Tool use is 2x more expensive but 100% reliable.

---

## How Tool Use Works

```
┌────────────────────────────────────────────────────────────────┐
│                        YOUR REQUEST                            │
│  ┌─────────────┐  ┌─────────────────────────────────────────┐  │
│  │   Prompt    │  │           Tool Schema (JSON)            │  │
│  │  "Analyze   │  │  {                                      │  │
│  │   this..."  │  │    "name": "analyze_review",            │  │
│  │             │  │    "input_schema": {                    │  │
│  │             │  │      "sentiment": {"enum": [...]},      │  │
│  │             │  │      "confidence": {"type": "number"}   │  │
│  │             │  │    }                                    │  │
│  │             │  │  }                                      │  │
│  └─────────────┘  └─────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              ↓
                    Claude API Processing
                              ↓
┌────────────────────────────────────────────────────────────────┐
│                      RESPONSE                                  │
│  {                                                             │
│    "content": [                                                │
│      {                                                         │
│        "type": "tool_use",                                     │
│        "name": "analyze_review",                               │
│        "input": {                    ← Pre-validated!          │
│          "sentiment": "positive",    ← Enum-constrained        │
│          "confidence": 0.9,          ← Type-checked            │
│          "key_points": [...]         ← Array validated         │
│        }                                                       │
│      }                                                         │
│    ]                                                           │
│  }                                                             │
└────────────────────────────────────────────────────────────────┘
```

The key insight: **validation happens inside the API**, not in your parsing code.

---

## When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Quick prototype | Raw text |
| Simple extraction | JSON mode |
| Production system | Tool use |
| Complex nested data | Tool use |
| Cost-sensitive | JSON mode |
| Mission-critical | Tool use |

---

## Running the Demo

```bash
cd /home/adam/LLMControl
PYTHONPATH=src/python python src/python/02_structured_outputs/demo.py --provider anthropic
```

**Expected output:**
```
============================================================
STRUCTURED OUTPUTS COMPARISON
Task: Extract structured review analysis
Reviews: 8
============================================================

APPROACH 1: Raw Text Parsing
  [✓] Review 1: parse=OK, sentiment=positive
  ...
  Parse Success Rate: 87.5%

APPROACH 2: JSON Mode
  [✓] Review 1: parse=OK, sentiment=positive
  ...
  Parse Success Rate: 87.5%

APPROACH 3: Tool Use (Function Calling)
  [✓] Review 1: parse=OK, sentiment=positive
      confidence=0.9, recommend=True
  ...
  Parse Success Rate: 100.0%
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `schemas.py` | Pydantic models, prompts, tool schema, test data |
| `parsers.py` | Three parsing implementations |
| `demo.py` | Comparison demo script |
| `README.md` | This documentation |

---

## Connection to Module 1

In Module 1 (Prompt Engineering), we saw that the original chain-of-thought prompt failed because it generated text containing "positive" and "negative" as analysis terms, confusing our parser.

**That was a parsing problem, not a model problem.**

Structured outputs solve this by eliminating ambiguous parsing entirely:
- Instead of searching for keywords, you get typed fields
- Instead of regex patterns, you get validated JSON
- Instead of hoping the model follows format, you enforce it

---

## Pydantic Integration

We use Pydantic for additional validation:

```python
from pydantic import BaseModel, Field
from typing import Literal, List

class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_points: List[str] = Field(min_length=1, max_length=5)
    recommendation: bool
    summary: str = Field(max_length=200)
```

This provides:
- Type checking (`confidence` must be a float)
- Value constraints (`confidence` between 0.0 and 1.0)
- Enum enforcement (`sentiment` must be one of four values)
- List length limits (`key_points` 1-5 items)

---

## Key Takeaways

1. **Start with JSON mode** - It's the best balance of simplicity and reliability

2. **Use tool_use for production** - When parsing failures have real consequences

3. **Always validate with Pydantic** - Even tool_use responses can have edge cases

4. **Schema design matters** - Unclear options (like missing "mixed") cause failures

5. **Measure parse success rate** - It's not just about accuracy, it's about reliability

---

## Next Steps

After mastering structured outputs, move to:
- **Module 3: Function Calling** - Using tools to take actions, not just extract data
- **Module 4: RAG** - Adding external knowledge to your prompts
