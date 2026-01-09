#!/usr/bin/env python3
"""
MCP Server for LLMControl.

Exposes RAG and Evaluation tools via the Model Context Protocol.

Usage:
    python server.py                    # Run server on stdio
    python server.py --port 8080        # Run on HTTP port

The server exposes these tools:
- rag_search: Search the knowledge base
- rag_query: Ask questions using RAG
- rag_stats: Get knowledge base statistics
- eval_exact_match: Check exact string matching
- eval_contains_keywords: Check keyword presence
- eval_semantic_similarity: Compare text similarity
- eval_llm_judge: LLM-based quality evaluation
- eval_faithfulness: Check for hallucinations
- eval_json: Validate JSON format
"""

import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import Any

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    ResourceTemplate,
)

from tools.rag_tool import RAGTool
from tools.eval_tool import EvalTool


# Initialize tools
rag_tool = RAGTool()
eval_tool = EvalTool()


# Create MCP server
server = Server("llmcontrol-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    tools = []

    # RAG tools
    for tool_def in RAGTool.get_tool_definitions():
        tools.append(Tool(
            name=tool_def["name"],
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        ))

    # Eval tools
    for tool_def in EvalTool.get_tool_definitions():
        tools.append(Tool(
            name=tool_def["name"],
            description=tool_def["description"],
            inputSchema=tool_def["inputSchema"]
        ))

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        result = await handle_tool_call(name, arguments)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str)
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, indent=2)
        )]


async def handle_tool_call(name: str, arguments: dict) -> dict:
    """Route tool calls to appropriate handlers."""

    # RAG tools
    if name == "rag_search":
        return rag_tool.search(
            query=arguments["query"],
            n_results=arguments.get("n_results", 3)
        )

    elif name == "rag_query":
        return rag_tool.query(question=arguments["question"])

    elif name == "rag_stats":
        return rag_tool.get_stats()

    # Eval tools
    elif name == "eval_exact_match":
        return eval_tool.exact_match(
            actual=arguments["actual"],
            expected=arguments["expected"],
            case_sensitive=arguments.get("case_sensitive", False)
        )

    elif name == "eval_contains_keywords":
        return eval_tool.contains_keywords(
            text=arguments["text"],
            keywords=arguments["keywords"]
        )

    elif name == "eval_semantic_similarity":
        return eval_tool.semantic_similarity(
            text1=arguments["text1"],
            text2=arguments["text2"],
            threshold=arguments.get("threshold", 0.7)
        )

    elif name == "eval_llm_judge":
        return eval_tool.llm_judge(
            output=arguments["output"],
            question=arguments.get("question"),
            expected=arguments.get("expected"),
            criteria=arguments.get("criteria")
        )

    elif name == "eval_faithfulness":
        return eval_tool.check_faithfulness(
            output=arguments["output"],
            context=arguments["context"]
        )

    elif name == "eval_json":
        return eval_tool.validate_json(text=arguments["text"])

    else:
        return {"error": f"Unknown tool: {name}"}


@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="llmcontrol://knowledge-base/stats",
            name="Knowledge Base Statistics",
            description="Statistics about the RAG knowledge base",
            mimeType="application/json"
        ),
        Resource(
            uri="llmcontrol://tools/list",
            name="Available Tools",
            description="List of all available MCP tools",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource."""
    if uri == "llmcontrol://knowledge-base/stats":
        stats = rag_tool.get_stats()
        return json.dumps(stats, indent=2)

    elif uri == "llmcontrol://tools/list":
        tools = []
        for tool_def in RAGTool.get_tool_definitions():
            tools.append({"category": "rag", **tool_def})
        for tool_def in EvalTool.get_tool_definitions():
            tools.append({"category": "eval", **tool_def})
        return json.dumps(tools, indent=2)

    else:
        return json.dumps({"error": f"Unknown resource: {uri}"})


async def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="LLMControl MCP Server")
    parser.add_argument("--port", type=int, help="HTTP port (if not using stdio)")
    args = parser.parse_args()

    print("Starting LLMControl MCP Server...", file=sys.stderr)
    print("Available tools:", file=sys.stderr)
    for tool in RAGTool.get_tool_definitions():
        print(f"  - {tool['name']}: {tool['description']}", file=sys.stderr)
    for tool in EvalTool.get_tool_definitions():
        print(f"  - {tool['name']}: {tool['description']}", file=sys.stderr)

    # Run with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
