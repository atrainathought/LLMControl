# Module 10: MCP Integration

Model Context Protocol integration for LLMControl - expose RAG and Evals as tools.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MCP ARCHITECTURE                            │
│                                                                      │
│   ┌─────────────────┐                                               │
│   │  Claude Desktop │                                               │
│   │  or MCP Client  │                                               │
│   └────────┬────────┘                                               │
│            │                                                         │
│            │ MCP Protocol (stdio/HTTP)                              │
│            │                                                         │
│            ▼                                                         │
│   ┌─────────────────────────────────────────────────┐               │
│   │              MCP SERVER (server.py)             │               │
│   │  ┌─────────────────────────────────────────┐    │               │
│   │  │           Tool Router                    │    │               │
│   │  │  rag_* → RAGTool   eval_* → EvalTool   │    │               │
│   │  └─────────────────────────────────────────┘    │               │
│   └─────────────────────────────────────────────────┘               │
│            │                       │                                 │
│            ▼                       ▼                                 │
│   ┌────────────────┐      ┌────────────────┐                        │
│   │   RAG Tools    │      │   Eval Tools   │                        │
│   │  (Module 9)    │      │  (Module 8)    │                        │
│   └────────────────┘      └────────────────┘                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Direct Tool Usage (No MCP)

```python
from tools.rag_tool import RAGTool
from tools.eval_tool import EvalTool

# RAG
rag = RAGTool()
result = rag.query("What is the leave policy?")
print(result["answer"])

# Evals
evals = EvalTool()
result = evals.llm_judge(
    output="The answer is 42",
    question="What is the meaning of life?"
)
print(result["score"])
```

### Option 2: MCP Server for Claude Desktop

```bash
# Start the server
cd src/python/10_mcp_integration
python server.py
```

Add to Claude Desktop config (`~/.config/claude/claude_desktop_config.json`):

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

### Option 3: Programmatic MCP Client

```python
from client import LLMControlClient

async with LLMControlClient().connect() as client:
    # List tools
    tools = await client.list_tools()

    # Use RAG
    result = await client.rag_query("How many days of leave?")

    # Use Evals
    result = await client.eval_semantic_similarity(
        "cat on mat",
        "feline on rug"
    )
```

---

## Available Tools

### RAG Tools

| Tool | Description |
|------|-------------|
| `rag_search` | Search the knowledge base for relevant documents |
| `rag_query` | Ask a question and get an answer using RAG |
| `rag_stats` | Get knowledge base statistics |

### Eval Tools

| Tool | Description |
|------|-------------|
| `eval_exact_match` | Check if two strings match exactly |
| `eval_contains_keywords` | Check if text contains all keywords |
| `eval_semantic_similarity` | Compare semantic similarity |
| `eval_llm_judge` | LLM-based quality evaluation |
| `eval_faithfulness` | Check for hallucinations |
| `eval_json` | Validate JSON format |

---

## Tool Schemas

### rag_query

```json
{
  "name": "rag_query",
  "description": "Ask a question and get an answer from the knowledge base",
  "inputSchema": {
    "type": "object",
    "properties": {
      "question": {
        "type": "string",
        "description": "The question to answer"
      }
    },
    "required": ["question"]
  }
}
```

**Response:**
```json
{
  "question": "How many days of annual leave?",
  "answer": "Employees receive 20 days of annual leave...",
  "confidence": 0.9,
  "success": true,
  "iterations": 1,
  "sources_used": 2,
  "total_tokens": 450
}
```

### eval_llm_judge

```json
{
  "name": "eval_llm_judge",
  "description": "Use LLM to judge output quality",
  "inputSchema": {
    "type": "object",
    "properties": {
      "output": {"type": "string"},
      "question": {"type": "string"},
      "expected": {"type": "string"},
      "criteria": {"type": "string"}
    },
    "required": ["output"]
  }
}
```

**Response:**
```json
{
  "evaluator": "llm_judge",
  "passed": true,
  "score": 0.85,
  "reasoning": "The response is accurate and well-structured...",
  "strengths": ["Clear", "Comprehensive"],
  "issues": []
}
```

---

## Files

```
10_mcp_integration/
├── __init__.py           # Package exports
├── server.py             # MCP server (stdio transport)
├── client.py             # MCP client for testing
├── demo.py               # Comprehensive demo
├── tools/
│   ├── __init__.py
│   ├── rag_tool.py       # RAG tool wrapper
│   └── eval_tool.py      # Eval tool wrapper
└── README.md             # This file
```

---

## Demo Output

```
======================================================================
DIRECT TOOL USAGE
======================================================================

--- RAG TOOLS ---

1. rag_search('password policy')
   Found 2 results
   - doc_002_chunk_0: 78.5% similarity
   - doc_002_chunk_1: 65.2% similarity

2. rag_query('What is the password expiration policy?')
   Answer: According to the IT Security Policy, passwords expire every 90 days...
   Confidence: 90%
   Iterations: 1

--- EVAL TOOLS ---

1. eval_exact_match('Paris', 'paris')
   Passed: True, Score: 1.0

2. eval_semantic_similarity('cat on mat', 'feline on rug')
   Passed: True, Score: 85.2%

3. eval_llm_judge('Exercise improves health...')
   Passed: True, Score: 88%
   Reasoning: The response is accurate and relevant...

4. eval_faithfulness(hallucinated_response)
   Score: 20%
   Hallucinations: ['2000 employees', 'founded in 2005']
```

---

## Claude Desktop Integration

Once configured, you can use these tools directly in Claude Desktop:

**User:** Search the knowledge base for information about expenses

**Claude:** I'll search the knowledge base for you.
```
Tool: rag_search
Arguments: {"query": "expense reimbursement policy"}
```

**Result:** Found 3 relevant documents about expense policies...

**User:** What's the daily meal limit for travel?

**Claude:** Let me query the RAG system.
```
Tool: rag_query
Arguments: {"question": "What is the daily meal expense limit for travel?"}
```

**Answer:** According to the Expense Reimbursement Policy, the daily meal limit is $75 for domestic travel and $100 for international travel.

---

## MCP Protocol Details

### Transport

The server uses **stdio** transport by default:
- Reads JSON-RPC requests from stdin
- Writes JSON-RPC responses to stdout
- Compatible with Claude Desktop and other MCP clients

### Message Flow

```
Client                              Server
  │                                    │
  │──── initialize ───────────────────▶│
  │◀─── initialized ──────────────────│
  │                                    │
  │──── tools/list ───────────────────▶│
  │◀─── tools (9 tools) ──────────────│
  │                                    │
  │──── tools/call (rag_query) ───────▶│
  │◀─── result (answer) ──────────────│
  │                                    │
```

### Error Handling

```json
{
  "error": "Tool 'unknown_tool' not found"
}
```

---

## Extending the Server

### Adding New Tools

1. Create a tool class in `tools/`:

```python
class MyTool:
    def my_function(self, arg: str) -> Dict:
        return {"result": arg}

    @staticmethod
    def get_tool_definitions() -> list:
        return [{
            "name": "my_function",
            "description": "Does something",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "arg": {"type": "string"}
                },
                "required": ["arg"]
            }
        }]
```

2. Register in `server.py`:

```python
from tools.my_tool import MyTool
my_tool = MyTool()

# Add to list_tools()
for tool_def in MyTool.get_tool_definitions():
    tools.append(Tool(...))

# Add to handle_tool_call()
if name == "my_function":
    return my_tool.my_function(arguments["arg"])
```

---

## Best Practices

### 1. Tool Naming

- Use prefixes: `rag_*`, `eval_*`, `my_*`
- Keep names short but descriptive
- Use snake_case

### 2. Input Validation

```python
async def handle_tool_call(name: str, arguments: dict) -> dict:
    try:
        # Validate required arguments
        if "query" not in arguments:
            return {"error": "Missing required argument: query"}

        return tool.function(arguments["query"])
    except Exception as e:
        return {"error": str(e)}
```

### 3. Response Format

Always return structured JSON:
```python
{
    "success": True,
    "result": {...},
    "metadata": {
        "latency_ms": 123,
        "tokens_used": 456
    }
}
```

---

## Troubleshooting

### Server won't start

```bash
# Check MCP is installed
pip show mcp

# Check Python version (3.10+ required)
python --version

# Run with verbose output
python server.py 2>&1 | head -20
```

### Claude Desktop not seeing tools

1. Check config path is absolute
2. Restart Claude Desktop
3. Check server logs in Claude Desktop developer tools

### Tool returns error

```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Summary

| Feature | Status |
|---------|--------|
| RAG Search | ✓ |
| RAG Query (Agentic) | ✓ |
| KB Statistics | ✓ |
| Exact Match Eval | ✓ |
| Keyword Eval | ✓ |
| Semantic Eval | ✓ |
| LLM Judge | ✓ |
| Faithfulness Check | ✓ |
| JSON Validation | ✓ |
| Claude Desktop Support | ✓ |
| Stdio Transport | ✓ |

This completes the LLMControl project with full MCP integration!
