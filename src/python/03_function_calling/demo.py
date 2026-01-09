#!/usr/bin/env python3
"""
Function Calling Demo

This script demonstrates how LLMs can use tools to:
1. Perform calculations accurately
2. Access external data (weather, database)
3. Chain multiple tools together
4. Compare with vs without tool access

Usage:
    python demo.py --provider anthropic
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_client import AnthropicClient
from shared.metrics import MetricsTracker

from tools import TEST_QUERIES, ALL_TOOLS
from executor import run_tool_loop, run_without_tools


def main():
    parser = argparse.ArgumentParser(description="Function Calling Demo")
    parser.add_argument(
        "--provider",
        choices=["anthropic"],
        default="anthropic",
        help="LLM provider to use",
    )
    args = parser.parse_args()

    # Initialize
    tracker = MetricsTracker("03_function_calling")

    try:
        client = AnthropicClient()
    except Exception as e:
        print(f"Error: Could not initialize Anthropic client: {e}")
        return 1

    print("\n" + "="*70)
    print("FUNCTION CALLING / TOOL USE DEMO")
    print("="*70)
    print(f"Available tools: {', '.join(t['name'] for t in ALL_TOOLS)}")
    print(f"Test queries: {len(TEST_QUERIES)}")
    print("="*70)

    # Run experiments with tools
    print("\n" + "-"*70)
    print("EXPERIMENT 1: With Tools")
    print("-"*70)

    with_tools_exp = tracker.create_experiment(
        experiment_name="function_calling",
        approach="with_tools",
        provider="anthropic",
        model=client.model,
    )

    for i, test in enumerate(TEST_QUERIES):
        print(f"\n{'='*70}")
        print(f"Query {i+1}: {test['query']}")
        print(f"Description: {test['description']}")
        print(f"Expected tools: {test['expected_tools']}")
        print("-"*70)

        result = run_tool_loop(client, test["query"], verbose=True)

        # Check if expected tools were called
        called_tools = [tc.tool_name for tc in result.tool_calls]
        tools_correct = set(called_tools) == set(test["expected_tools"]) or \
                       all(t in called_tools for t in test["expected_tools"])

        with_tools_exp.add_sample(
            correct=tools_correct and result.success,
            latency_ms=result.total_latency_ms,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            cost_usd=0,  # Calculate separately if needed
        )

        print(f"\n  Tools called: {called_tools}")
        print(f"  Turns: {result.total_turns}")
        print(f"  Total latency: {result.total_latency_ms:.0f}ms")
        print(f"  Final response: {result.final_response[:200]}...")

    # Run experiments without tools
    print("\n" + "-"*70)
    print("EXPERIMENT 2: Without Tools (Baseline)")
    print("-"*70)

    without_tools_exp = tracker.create_experiment(
        experiment_name="function_calling",
        approach="without_tools",
        provider="anthropic",
        model=client.model,
    )

    # Just run a few representative queries
    baseline_queries = [
        TEST_QUERIES[0],  # Math calculation
        TEST_QUERIES[1],  # Weather
        TEST_QUERIES[2],  # Database query
    ]

    for i, test in enumerate(baseline_queries):
        print(f"\n{'='*70}")
        print(f"Query: {test['query']}")
        print("-"*70)

        result = run_without_tools(client, test["query"], verbose=True)

        without_tools_exp.add_sample(
            correct=False,  # Without tools, answers are likely wrong/incomplete
            latency_ms=result.total_latency_ms,
            input_tokens=result.total_input_tokens,
            output_tokens=result.total_output_tokens,
            cost_usd=0,
        )

        print(f"\n  Response: {result.final_response[:300]}...")

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(tracker.compare(baseline_approach="without_tools"))

    # Save results
    tracker.save()

    # Print key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
1. WITH TOOLS:
   - LLM correctly identifies which tools to use
   - Calculator provides precise mathematical results
   - Weather/DB queries return real (simulated) data
   - Multi-step queries handled via conversation turns

2. WITHOUT TOOLS:
   - Math: LLM may calculate correctly but can make errors
   - Weather: LLM admits it doesn't have current data
   - Database: LLM cannot access product information

3. TOOL CALLING PATTERN:
   User Query → LLM decides tool → Execute → Result → LLM synthesizes
                    ↑                                      ↓
                    └──────── Loop if more tools needed ───┘
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
