"""
MCP Integration Demo Page
"""

import streamlit as st
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(page_title="MCP Integration | LLMControl", page_icon="🔌", layout="wide")

apply_styles()
page_header("MCP Integration", "Model Context Protocol - Connect LLMs to tool ecosystems", "🔌")

section_divider()

# Explanation
with st.expander("What is MCP?", expanded=False):
    st.markdown("""
    **MCP (Model Context Protocol)** is Anthropic's standard for connecting LLMs to external tools.

    ```
    Client (Claude Desktop) ←→ MCP Server ←→ Tools (RAG, Evals, APIs)
    ```

    | Component | Role |
    |-----------|------|
    | **MCP Client** | Claude Desktop, IDE extensions |
    | **MCP Server** | Exposes tools via JSON-RPC |
    | **Tools** | Functions the LLM can call |
    | **Resources** | Data the LLM can access |
    """)

# Server Overview
st.markdown("### LLMControl MCP Server")

st.markdown("Our MCP server exposes RAG and Evals capabilities as tools:")

# Available Tools
tools = {
    "RAG Tools": [
        {"name": "rag_search", "description": "Search knowledge base", "params": "query, top_k"},
        {"name": "rag_query", "description": "Search + generate answer", "params": "question"},
        {"name": "rag_stats", "description": "Get index statistics", "params": "none"},
    ],
    "Eval Tools": [
        {"name": "eval_exact_match", "description": "Check exact string match", "params": "actual, expected"},
        {"name": "eval_contains", "description": "Check keyword presence", "params": "text, keywords"},
        {"name": "eval_semantic", "description": "Semantic similarity", "params": "text1, text2"},
        {"name": "eval_llm_judge", "description": "LLM quality assessment", "params": "output, question"},
        {"name": "eval_faithfulness", "description": "Check hallucinations", "params": "output, context"},
        {"name": "eval_json_schema", "description": "Validate JSON structure", "params": "data, schema"},
    ]
}

col1, col2 = st.columns(2)

for i, (category, tool_list) in enumerate(tools.items()):
    with col1 if i == 0 else col2:
        st.markdown(f"**{category}**")
        for tool in tool_list:
            with st.expander(tool["name"]):
                st.write(tool["description"])
                st.caption(f"Parameters: {tool['params']}")

section_divider()

# Interactive Demo
st.markdown("### Interactive Demo")

selected_tool = st.selectbox("Select tool", [
    "rag_search", "rag_query", "eval_exact_match",
    "eval_contains", "eval_semantic", "eval_faithfulness"
])

# Tool-specific inputs
if selected_tool == "rag_search":
    query = st.text_input("Search query", "annual leave policy")
    top_k = st.slider("Results", 1, 10, 3)
    params = {"query": query, "top_k": top_k}

elif selected_tool == "rag_query":
    question = st.text_input("Question", "How many days of annual leave?")
    params = {"question": question}

elif selected_tool == "eval_exact_match":
    actual = st.text_input("Actual", "Paris")
    expected = st.text_input("Expected", "paris")
    params = {"actual": actual, "expected": expected}

elif selected_tool == "eval_contains":
    text = st.text_area("Text", "Python is great for ML and AI", height=80)
    keywords = st.text_input("Keywords (comma-separated)", "Python, ML")
    params = {"text": text, "keywords": keywords.split(",")}

elif selected_tool == "eval_semantic":
    text1 = st.text_input("Text 1", "The cat sat on the mat")
    text2 = st.text_input("Text 2", "A feline rested on the rug")
    params = {"text1": text1, "text2": text2}

else:  # eval_faithfulness
    output = st.text_area("Output", "TechFlow has 500 employees", height=80)
    context = st.text_area("Context", "TechFlow Inc has 500 employees and $150M revenue", height=80)
    params = {"output": output, "context": context}

if st.button("Call Tool", type="primary", use_container_width=True):
    section_divider()
    st.markdown("### Request")

    request = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": selected_tool,
            "arguments": params
        },
        "id": 1
    }

    st.code(json.dumps(request, indent=2), language="json")

    st.markdown("### Response")

    # Simulated responses
    responses = {
        "rag_search": {
            "results": [
                {"content": "All employees get 20 days annual leave...", "score": 0.92},
                {"content": "Leave accrues at 1.67 days per month...", "score": 0.85},
            ],
            "count": 2
        },
        "rag_query": {
            "answer": "Employees receive 20 days of annual leave per year.",
            "sources": 2,
            "confidence": 0.95
        },
        "eval_exact_match": {
            "passed": True,
            "score": 1.0,
            "case_insensitive": True
        },
        "eval_contains": {
            "passed": True,
            "score": 1.0,
            "found": ["Python", "ML"],
            "missing": []
        },
        "eval_semantic": {
            "passed": True,
            "score": 0.87,
            "threshold": 0.7
        },
        "eval_faithfulness": {
            "passed": True,
            "score": 1.0,
            "supported_claims": ["TechFlow has 500 employees"],
            "hallucinations": []
        }
    }

    response = {
        "jsonrpc": "2.0",
        "result": responses.get(selected_tool, {"status": "ok"}),
        "id": 1
    }

    st.code(json.dumps(response, indent=2), language="json")

    # Summary
    result = responses.get(selected_tool, {})
    if "passed" in result:
        if result["passed"]:
            st.success(f"**PASSED** - Score: {result.get('score', 1.0):.0%}")
        else:
            st.error(f"**FAILED** - Score: {result.get('score', 0):.0%}")
    elif "answer" in result:
        st.success(f"**Answer:** {result['answer']}")

section_divider()

# Setup Instructions
st.markdown("### Setup Instructions")

tab1, tab2 = st.tabs(["Claude Desktop", "Custom Client"])

with tab1:
    st.markdown("**1. Install the MCP server**")
    st.code("pip install llmcontrol-mcp", language="bash")

    st.markdown("**2. Add to Claude Desktop config**")
    st.code("""
# ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "llmcontrol": {
      "command": "python",
      "args": ["-m", "llmcontrol.mcp.server"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key"
      }
    }
  }
}
    """, language="json")

    st.markdown("**3. Restart Claude Desktop**")
    st.info("The tools will appear in Claude's tool picker")

with tab2:
    st.markdown("**Connect programmatically**")
    st.code("""
from mcp import Client
import asyncio

async def main():
    # Connect to MCP server
    client = Client()
    await client.connect_stdio(
        "python", ["-m", "llmcontrol.mcp.server"]
    )

    # List available tools
    tools = await client.list_tools()
    print(f"Available: {[t.name for t in tools]}")

    # Call a tool
    result = await client.call_tool(
        "rag_query",
        {"question": "What is the leave policy?"}
    )
    print(f"Answer: {result}")

asyncio.run(main())
    """, language="python")

section_divider()

# Architecture
st.markdown("### Architecture")

st.code("""
┌─────────────────┐     JSON-RPC      ┌─────────────────┐
│  Claude Desktop │ ←──────────────── │   MCP Server    │
│  (MCP Client)   │                   │  (llmcontrol)   │
└─────────────────┘                   └────────┬────────┘
                                               │
                                               ▼
                                    ┌─────────────────────┐
                                    │      Tools          │
                                    ├─────────┬───────────┤
                                    │   RAG   │   Evals   │
                                    │ Search  │  Judge    │
                                    │ Query   │  Semantic │
                                    │ Stats   │  Match    │
                                    └─────────┴───────────┘
""", language="text")

# Code example
with st.expander("Server Implementation"):
    st.code("""
from mcp.server import Server
from mcp.types import Tool

server = Server("llmcontrol")

@server.tool()
async def rag_search(query: str, top_k: int = 3) -> dict:
    \"\"\"Search the knowledge base.\"\"\"
    results = vectorstore.search(query, n_results=top_k)
    return {
        "results": [
            {"content": r["content"], "score": r["similarity"]}
            for r in results
        ]
    }

@server.tool()
async def eval_semantic(text1: str, text2: str) -> dict:
    \"\"\"Compare semantic similarity.\"\"\"
    evaluator = SemanticSimilarityEvaluator()
    result = evaluator.evaluate(text1, text2)
    return {
        "passed": result.passed,
        "score": result.score
    }

# Run server
if __name__ == "__main__":
    server.run_stdio()
    """, language="python")

setup_sidebar()
