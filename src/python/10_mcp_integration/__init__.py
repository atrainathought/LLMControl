"""
MCP Integration Module.

Exposes LLMControl tools via Model Context Protocol (MCP).

Usage:
    # Direct tool usage
    from 10_mcp_integration.tools import RAGTool, EvalTool

    rag = RAGTool()
    result = rag.query("What is the leave policy?")

    # MCP server
    python server.py

    # MCP client
    from 10_mcp_integration.client import LLMControlClient
    async with client.connect():
        result = await client.rag_query("...")
"""

from .tools import RAGTool, EvalTool

__all__ = ["RAGTool", "EvalTool"]
