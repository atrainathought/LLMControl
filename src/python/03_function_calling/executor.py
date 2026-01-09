"""
Tool Execution Loop

This module implements the agentic loop for tool use:
1. Send user query + available tools to LLM
2. LLM decides which tool(s) to call
3. Execute tools and collect results
4. Send results back to LLM
5. LLM synthesizes final answer
6. Repeat if more tools needed

Key concepts:
- Tool choice: LLM decides when to use tools
- Multi-turn: Results feed back into conversation
- Termination: Loop ends when LLM responds without tool calls
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from tools import ALL_TOOLS, execute_tool


@dataclass
class ToolCall:
    """Record of a single tool invocation."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_result: Dict[str, Any]
    latency_ms: float


@dataclass
class ExecutionResult:
    """Complete result of an agentic tool-use session."""
    query: str
    final_response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    total_turns: int = 0
    total_latency_ms: float = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


def run_tool_loop(
    client,  # AnthropicClient
    query: str,
    tools: List[Dict] = None,
    max_turns: int = 10,
    verbose: bool = True
) -> ExecutionResult:
    """
    Run the agentic tool execution loop.

    Args:
        client: Anthropic client instance
        query: User's question
        tools: List of tool schemas (defaults to ALL_TOOLS)
        max_turns: Maximum conversation turns to prevent infinite loops
        verbose: Print progress information

    Returns:
        ExecutionResult with final response and tool call history
    """
    if tools is None:
        tools = ALL_TOOLS

    result = ExecutionResult(query=query)

    # Initialize conversation
    messages = [{"role": "user", "content": query}]

    system_prompt = """You are a helpful assistant with access to tools.
Use the available tools to answer the user's questions accurately.
For calculations, always use the calculator tool - do not calculate in your head.
For weather, use the get_weather tool.
For product queries, use the query_products tool.
If a task requires multiple steps, call tools in sequence."""

    turn = 0
    while turn < max_turns:
        turn += 1
        result.total_turns = turn

        if verbose:
            print(f"\n  Turn {turn}:")

        # Call Claude with tools
        start_time = time.perf_counter()
        response = client.client.messages.create(
            model=client.model,
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        result.total_latency_ms += latency_ms
        result.total_input_tokens += response.usage.input_tokens
        result.total_output_tokens += response.usage.output_tokens

        # Check what Claude returned
        has_tool_use = False
        text_response = ""
        tool_uses = []

        for block in response.content:
            if block.type == "text":
                text_response = block.text
            elif block.type == "tool_use":
                has_tool_use = True
                tool_uses.append(block)

        # If no tool calls, we're done
        if not has_tool_use:
            result.final_response = text_response
            if verbose:
                print(f"    Final response (no more tools needed)")
            break

        # Process tool calls
        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            tool_id = tool_use.id

            if verbose:
                print(f"    Calling tool: {tool_name}")
                print(f"      Input: {json.dumps(tool_input, indent=2)[:100]}...")

            # Execute the tool
            tool_start = time.perf_counter()
            tool_result = execute_tool(tool_name, tool_input)
            tool_latency = (time.perf_counter() - tool_start) * 1000

            if verbose:
                print(f"      Result: {json.dumps(tool_result)[:100]}...")

            # Record the tool call
            result.tool_calls.append(ToolCall(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=tool_result,
                latency_ms=tool_latency
            ))

            # Format result for Claude
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": json.dumps(tool_result)
            })

        # Add assistant's response and tool results to conversation
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    # Check if we hit max turns
    if turn >= max_turns and not result.final_response:
        result.success = False
        result.error = f"Max turns ({max_turns}) reached without final response"
        result.final_response = text_response or "Unable to complete the request."

    return result


def run_without_tools(client, query: str, verbose: bool = True) -> ExecutionResult:
    """
    Run the same query WITHOUT tools for comparison.

    This demonstrates what happens when the LLM has to answer
    without external tool access.
    """
    result = ExecutionResult(query=query)

    system_prompt = """You are a helpful assistant. Answer the user's question to the best of your ability.
For calculations, compute the answer yourself.
For weather, provide your best estimate or say you don't have current data.
For product queries, explain that you don't have access to the database."""

    start_time = time.perf_counter()
    response = client.client.messages.create(
        model=client.model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": query}],
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    result.total_turns = 1
    result.total_latency_ms = latency_ms
    result.total_input_tokens = response.usage.input_tokens
    result.total_output_tokens = response.usage.output_tokens
    result.final_response = response.content[0].text

    if verbose:
        print(f"    Response (no tools): {result.final_response[:100]}...")

    return result
