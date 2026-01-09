#!/usr/bin/env python3
"""
MCP Integration Demo.

Demonstrates two ways to use the MCP tools:
1. Direct tool usage (without MCP protocol)
2. MCP client-server communication

Run this demo to see both approaches.
"""

import sys
import json
import asyncio
from pathlib import Path

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.rag_tool import RAGTool
from tools.eval_tool import EvalTool


def demo_direct_usage():
    """Demonstrate direct tool usage (no MCP protocol)."""
    print("=" * 70)
    print("DIRECT TOOL USAGE (No MCP Protocol)")
    print("=" * 70)
    print("\nThis shows using the tools directly without MCP.\n")

    # Initialize tools
    print("Initializing tools...")
    rag = RAGTool(use_agentic=True)
    evals = EvalTool()

    # RAG Demo
    print("\n" + "-" * 40)
    print("RAG TOOLS")
    print("-" * 40)

    # Search
    print("\n1. rag_search('password policy')")
    result = rag.search("password policy", n_results=2)
    print(f"   Found {len(result['results'])} results")
    for r in result["results"]:
        print(f"   - {r['id']}: {r['similarity']:.2%} similarity")
        print(f"     '{r['content'][:80]}...'")

    # Query
    print("\n2. rag_query('What is the password expiration policy?')")
    result = rag.query("What is the password expiration policy?")
    print(f"   Answer: {result['answer'][:150]}...")
    print(f"   Confidence: {result['confidence']:.0%}")
    print(f"   Iterations: {result['iterations']}")
    print(f"   Success: {result['success']}")

    # Stats
    print("\n3. rag_stats()")
    result = rag.get_stats()
    print(f"   Documents: {result['documents_count']}")
    print(f"   RAG Type: {result['rag_type']}")
    print(f"   Model: {result['model']}")

    # Eval Demo
    print("\n" + "-" * 40)
    print("EVAL TOOLS")
    print("-" * 40)

    # Exact match
    print("\n1. eval_exact_match('Paris', 'paris')")
    result = evals.exact_match("Paris", "paris", case_sensitive=False)
    print(f"   Passed: {result['passed']}, Score: {result['score']}")

    # Keywords
    print("\n2. eval_contains_keywords('Python is great for AI', ['Python', 'AI', 'ML'])")
    result = evals.contains_keywords("Python is great for AI", ["Python", "AI", "ML"])
    print(f"   Passed: {result['passed']}, Score: {result['score']:.2%}")
    print(f"   Found: {result['found']}, Missing: {result['missing']}")

    # Semantic similarity
    print("\n3. eval_semantic_similarity('cat on mat', 'feline on rug')")
    result = evals.semantic_similarity("The cat sat on the mat", "A feline rested on a rug")
    print(f"   Passed: {result['passed']}, Score: {result['score']:.2%}")

    # LLM judge
    print("\n4. eval_llm_judge('Exercise improves health...')")
    result = evals.llm_judge(
        output="Regular exercise improves cardiovascular health and mental well-being.",
        question="What are the benefits of exercise?"
    )
    print(f"   Passed: {result['passed']}, Score: {result['score']:.0%}")
    print(f"   Reasoning: {result['reasoning'][:100]}...")

    # Faithfulness
    print("\n5. eval_faithfulness(output, context)")
    context = "TechFlow was founded in 2010 and has 500 employees."
    faithful = "TechFlow has 500 employees."
    hallucinated = "TechFlow has 2000 employees and was founded in 2005."

    result = evals.check_faithfulness(faithful, context)
    print(f"   Faithful response: score={result['score']:.0%}")

    result = evals.check_faithfulness(hallucinated, context)
    print(f"   Hallucinated response: score={result['score']:.0%}")
    print(f"   Hallucinations: {result['hallucinations']}")

    # JSON validation
    print("\n6. eval_json('{\"name\": \"Alice\", \"age\": 30}')")
    result = evals.validate_json('{"name": "Alice", "age": 30}')
    print(f"   Valid: {result['passed']}")

    result = evals.validate_json('{"name": "Bob", age: 25}')  # Invalid
    print(f"   Invalid JSON: passed={result['passed']}")


async def demo_mcp_protocol():
    """Demonstrate MCP client-server communication."""
    print("\n\n" + "=" * 70)
    print("MCP PROTOCOL USAGE (Client-Server)")
    print("=" * 70)
    print("\nThis shows using tools via MCP protocol.\n")

    from client import LLMControlClient

    client = LLMControlClient()

    try:
        async with client.connect():
            # List tools
            print("-" * 40)
            print("AVAILABLE MCP TOOLS")
            print("-" * 40)
            tools = await client.list_tools()
            print(f"\nFound {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}")

            # RAG query via MCP
            print("\n" + "-" * 40)
            print("RAG QUERY VIA MCP")
            print("-" * 40)

            result = await client.rag_query("How many days of sick leave do employees get?")
            print(f"\nQuestion: How many days of sick leave...")
            print(f"Answer: {result.get('answer', 'N/A')[:200]}...")
            print(f"Confidence: {result.get('confidence', 0):.0%}")

            # Eval via MCP
            print("\n" + "-" * 40)
            print("EVAL VIA MCP")
            print("-" * 40)

            result = await client.eval_semantic_similarity(
                "Annual leave is 20 days",
                "Employees get twenty days vacation"
            )
            print(f"\nSemantic similarity: {result.get('score', 0):.2%}")

            print("\n" + "-" * 40)
            print("MCP DEMO COMPLETE")
            print("-" * 40)

    except Exception as e:
        print(f"\nMCP connection error: {e}")
        print("(This is expected if running demo.py directly)")
        print("To test MCP protocol, run: python client.py")


def print_tool_definitions():
    """Print all tool definitions for reference."""
    print("\n\n" + "=" * 70)
    print("MCP TOOL DEFINITIONS")
    print("=" * 70)

    print("\n--- RAG Tools ---")
    for tool in RAGTool.get_tool_definitions():
        print(f"\n{tool['name']}:")
        print(f"  Description: {tool['description']}")
        print(f"  Input Schema: {json.dumps(tool['inputSchema'], indent=4)}")

    print("\n--- Eval Tools ---")
    for tool in EvalTool.get_tool_definitions():
        print(f"\n{tool['name']}:")
        print(f"  Description: {tool['description']}")
        props = tool['inputSchema'].get('properties', {})
        print(f"  Parameters: {list(props.keys())}")


def print_summary():
    """Print demo summary."""
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
MCP INTEGRATION MODULE

This module exposes LLMControl tools via Model Context Protocol (MCP).

ARCHITECTURE:
┌─────────────────────────────────────────────────────────────────┐
│                         MCP CLIENT                              │
│  (Claude Desktop, custom apps, other MCP-compatible clients)   │
└─────────────────────────┬───────────────────────────────────────┘
                          │ MCP Protocol (stdio/HTTP)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MCP SERVER                              │
│  server.py - Handles tool discovery and execution              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐            ┌──────────────────┐
│    RAG Tools     │            │   Eval Tools     │
│  - rag_search    │            │  - eval_exact    │
│  - rag_query     │            │  - eval_keywords │
│  - rag_stats     │            │  - eval_semantic │
│                  │            │  - eval_judge    │
│  (Module 9)      │            │  - eval_faithful │
│  Agentic RAG     │            │  (Module 8)      │
└──────────────────┘            └──────────────────┘

USAGE OPTIONS:

1. DIRECT USAGE (No MCP):
   from tools.rag_tool import RAGTool
   rag = RAGTool()
   result = rag.query("What is the leave policy?")

2. MCP SERVER (For Claude Desktop):
   python server.py
   # Then connect from Claude Desktop

3. MCP CLIENT (Programmatic):
   from client import LLMControlClient
   async with client.connect():
       result = await client.rag_query("...")

CLAUDE DESKTOP CONFIG:
Add to claude_desktop_config.json:
{
  "mcpServers": {
    "llmcontrol": {
      "command": "python",
      "args": ["/path/to/10_mcp_integration/server.py"]
    }
  }
}
""")


def main():
    """Run the complete demo."""
    print("=" * 70)
    print("MCP INTEGRATION DEMO")
    print("=" * 70)
    print("""
This module provides MCP (Model Context Protocol) integration for LLMControl.

MCP allows Claude Desktop and other compatible clients to use our
RAG and Evaluation tools as if they were native capabilities.

The demo will show:
1. Direct tool usage (without MCP)
2. Tool definitions for MCP
3. Summary of the architecture
""")

    # Direct usage demo
    demo_direct_usage()

    # Tool definitions
    print_tool_definitions()

    # Summary
    print_summary()

    # Note about MCP protocol demo
    print("\n" + "=" * 70)
    print("NOTE: MCP Protocol Demo")
    print("=" * 70)
    print("""
To test the full MCP client-server communication:

1. Run the server:
   python server.py

2. In another terminal, run the client:
   python client.py

Or configure Claude Desktop to use the server.
""")


if __name__ == "__main__":
    main()
