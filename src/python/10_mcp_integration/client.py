#!/usr/bin/env python3
"""
MCP Client for LLMControl.

Demonstrates connecting to the MCP server and using tools.

Usage:
    # Start server in one terminal:
    python server.py

    # Run client in another:
    python client.py
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List
from contextlib import asynccontextmanager

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class LLMControlClient:
    """
    Client for the LLMControl MCP server.
    """

    def __init__(self):
        self.session: ClientSession = None
        self.tools: List[Dict] = []

    @asynccontextmanager
    async def connect(self, server_script: str = None):
        """
        Connect to the MCP server.

        Args:
            server_script: Path to server.py (defaults to same directory)
        """
        if server_script is None:
            server_script = str(Path(__file__).parent / "server.py")

        server_params = StdioServerParameters(
            command="python",
            args=[server_script],
            env=None
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session

                # Initialize
                await session.initialize()

                # Get available tools
                tools_response = await session.list_tools()
                self.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "schema": tool.inputSchema
                    }
                    for tool in tools_response.tools
                ]

                yield self

    async def list_tools(self) -> List[Dict]:
        """List available tools."""
        return self.tools

    async def call_tool(self, name: str, arguments: dict = None) -> Dict:
        """
        Call a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if arguments is None:
            arguments = {}

        result = await self.session.call_tool(name, arguments)

        # Parse the text content
        if result.content and len(result.content) > 0:
            text = result.content[0].text
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}

        return {"error": "No content returned"}

    # Convenience methods for RAG tools

    async def rag_search(self, query: str, n_results: int = 3) -> Dict:
        """Search the knowledge base."""
        return await self.call_tool("rag_search", {
            "query": query,
            "n_results": n_results
        })

    async def rag_query(self, question: str) -> Dict:
        """Ask a question using RAG."""
        return await self.call_tool("rag_query", {
            "question": question
        })

    async def rag_stats(self) -> Dict:
        """Get knowledge base statistics."""
        return await self.call_tool("rag_stats", {})

    # Convenience methods for Eval tools

    async def eval_exact_match(self, actual: str, expected: str, case_sensitive: bool = False) -> Dict:
        """Check exact match."""
        return await self.call_tool("eval_exact_match", {
            "actual": actual,
            "expected": expected,
            "case_sensitive": case_sensitive
        })

    async def eval_contains_keywords(self, text: str, keywords: List[str]) -> Dict:
        """Check keyword presence."""
        return await self.call_tool("eval_contains_keywords", {
            "text": text,
            "keywords": keywords
        })

    async def eval_semantic_similarity(self, text1: str, text2: str, threshold: float = 0.7) -> Dict:
        """Compare semantic similarity."""
        return await self.call_tool("eval_semantic_similarity", {
            "text1": text1,
            "text2": text2,
            "threshold": threshold
        })

    async def eval_llm_judge(self, output: str, question: str = None, expected: str = None) -> Dict:
        """LLM-based quality evaluation."""
        args = {"output": output}
        if question:
            args["question"] = question
        if expected:
            args["expected"] = expected
        return await self.call_tool("eval_llm_judge", args)

    async def eval_faithfulness(self, output: str, context: str) -> Dict:
        """Check for hallucinations."""
        return await self.call_tool("eval_faithfulness", {
            "output": output,
            "context": context
        })

    async def eval_json(self, text: str) -> Dict:
        """Validate JSON."""
        return await self.call_tool("eval_json", {"text": text})


async def demo():
    """Demonstrate the MCP client."""
    print("=" * 70)
    print("MCP CLIENT DEMO")
    print("=" * 70)

    client = LLMControlClient()

    async with client.connect():
        # List tools
        print("\n--- Available Tools ---")
        tools = await client.list_tools()
        for tool in tools:
            print(f"  {tool['name']}: {tool['description'][:50]}...")

        # RAG demos
        print("\n" + "=" * 70)
        print("RAG TOOLS")
        print("=" * 70)

        # Search
        print("\n--- rag_search ---")
        result = await client.rag_search("annual leave policy")
        print(f"Query: annual leave policy")
        print(f"Results: {len(result.get('results', []))} documents")
        for r in result.get("results", [])[:2]:
            print(f"  - {r['id']}: similarity={r['similarity']:.2%}")

        # Query
        print("\n--- rag_query ---")
        result = await client.rag_query("How many days of annual leave do employees get?")
        print(f"Question: How many days of annual leave...")
        print(f"Answer: {result.get('answer', 'N/A')[:150]}...")
        print(f"Confidence: {result.get('confidence', 0):.0%}")
        print(f"Iterations: {result.get('iterations', 0)}")

        # Stats
        print("\n--- rag_stats ---")
        result = await client.rag_stats()
        print(f"Documents: {result.get('documents_count', 0)}")
        print(f"RAG Type: {result.get('rag_type', 'unknown')}")

        # Eval demos
        print("\n" + "=" * 70)
        print("EVAL TOOLS")
        print("=" * 70)

        # Exact match
        print("\n--- eval_exact_match ---")
        result = await client.eval_exact_match("Paris", "paris")
        print(f"'Paris' vs 'paris': passed={result.get('passed')}, score={result.get('score')}")

        # Keywords
        print("\n--- eval_contains_keywords ---")
        result = await client.eval_contains_keywords(
            "Python is great for machine learning and AI",
            ["Python", "machine learning", "AI"]
        )
        print(f"Keywords check: passed={result.get('passed')}, found={result.get('found')}")

        # Semantic similarity
        print("\n--- eval_semantic_similarity ---")
        result = await client.eval_semantic_similarity(
            "The cat sat on the mat",
            "A feline rested on a rug"
        )
        print(f"Semantic similarity: score={result.get('score'):.2%}")

        # LLM judge
        print("\n--- eval_llm_judge ---")
        result = await client.eval_llm_judge(
            output="Regular exercise improves cardiovascular health, boosts mood, and helps with weight management.",
            question="What are the benefits of exercise?"
        )
        print(f"LLM Judge: score={result.get('score'):.0%}")
        print(f"Reasoning: {result.get('reasoning', 'N/A')[:100]}...")

        # JSON validation
        print("\n--- eval_json ---")
        result = await client.eval_json('{"name": "Alice", "age": 30}')
        print(f"JSON valid: passed={result.get('passed')}")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(demo())
