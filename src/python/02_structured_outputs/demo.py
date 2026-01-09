#!/usr/bin/env python3
"""
Structured Outputs Demo

This script demonstrates three approaches to getting structured data from LLMs:
1. Raw text parsing (fragile)
2. JSON mode (better)
3. Tool use / function calling (best)

The key insight: LLMs are unreliable text generators. The closer you can get
to schema-enforced outputs, the more reliable your system becomes.

Usage:
    python demo.py --provider anthropic
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_client import AnthropicClient
from shared.metrics import MetricsTracker

from schemas import (
    TEST_REVIEWS,
    RAW_TEXT_PROMPT,
    JSON_MODE_PROMPT,
    TOOL_USE_PROMPT,
    get_tool_schema,
)
from parsers import (
    parse_raw_text,
    parse_json_response,
    parse_tool_response,
)


def run_raw_text_experiment(client, reviews, tracker):
    """Run experiment with raw text parsing."""
    print("\n" + "="*60)
    print("APPROACH 1: Raw Text Parsing")
    print("="*60)
    print("Strategy: Ask for analysis, parse with regex")
    print("-"*60)

    experiment = tracker.create_experiment(
        experiment_name="structured_outputs",
        approach="raw_text",
        provider="anthropic",
        model=client.model,
    )

    for i, item in enumerate(reviews):
        prompt = RAW_TEXT_PROMPT.format(review=item["review"])
        response = client.complete(prompt, temperature=0.0)

        result = parse_raw_text(response.content)

        # Check if sentiment matches expected
        correct = False
        if result["success"] and result["data"]:
            correct = result["data"].get("sentiment") == item["expected_sentiment"]

        experiment.add_sample(
            correct=correct,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

        status = "✓" if result["success"] else "✗"
        sentiment = result["data"].get("sentiment", "N/A") if result["data"] else "N/A"
        print(f"  [{status}] Review {i+1}: parse={'OK' if result['success'] else 'FAIL'}, sentiment={sentiment}")

        if not result["success"]:
            print(f"      Error: {result['error']}")

    print(f"\n  Parse Success Rate: {experiment.accuracy:.1%}")
    print(f"  Avg Latency: {experiment.latency_ms:.0f}ms")
    print(f"  Total Cost: ${experiment.cost_usd:.6f}")


def run_json_mode_experiment(client, reviews, tracker):
    """Run experiment with JSON mode."""
    print("\n" + "="*60)
    print("APPROACH 2: JSON Mode")
    print("="*60)
    print("Strategy: Ask for JSON, parse with json.loads()")
    print("-"*60)

    experiment = tracker.create_experiment(
        experiment_name="structured_outputs",
        approach="json_mode",
        provider="anthropic",
        model=client.model,
    )

    for i, item in enumerate(reviews):
        prompt = JSON_MODE_PROMPT.format(review=item["review"])
        response = client.complete(prompt, temperature=0.0)

        result = parse_json_response(response.content)

        # Check if sentiment matches expected
        correct = False
        if result["success"] and result["data"]:
            correct = result["data"].get("sentiment") == item["expected_sentiment"]

        experiment.add_sample(
            correct=correct,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

        status = "✓" if result["success"] else "✗"
        sentiment = result["data"].get("sentiment", "N/A") if result["data"] else "N/A"
        print(f"  [{status}] Review {i+1}: parse={'OK' if result['success'] else 'FAIL'}, sentiment={sentiment}")

        if not result["success"]:
            print(f"      Error: {result['error'][:80]}...")

    print(f"\n  Parse Success Rate: {experiment.accuracy:.1%}")
    print(f"  Avg Latency: {experiment.latency_ms:.0f}ms")
    print(f"  Total Cost: ${experiment.cost_usd:.6f}")


def run_tool_use_experiment(client, reviews, tracker):
    """Run experiment with tool use (best approach)."""
    print("\n" + "="*60)
    print("APPROACH 3: Tool Use (Function Calling)")
    print("="*60)
    print("Strategy: Define schema, Claude returns structured tool calls")
    print("-"*60)

    experiment = tracker.create_experiment(
        experiment_name="structured_outputs",
        approach="tool_use",
        provider="anthropic",
        model=client.model,
    )

    tool_schema = get_tool_schema()

    for i, item in enumerate(reviews):
        prompt = TOOL_USE_PROMPT.format(review=item["review"])

        # Make API call with tool
        import time
        start_time = time.perf_counter()

        response = client.client.messages.create(
            model=client.model,
            max_tokens=1024,
            temperature=0.0,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": "analyze_review"},
            messages=[{"role": "user", "content": prompt}],
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        result = parse_tool_response(response)

        # Check if sentiment matches expected
        correct = False
        if result["success"] and result["data"]:
            correct = result["data"].get("sentiment") == item["expected_sentiment"]

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        pricing = client.PRICING.get(client.model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        experiment.add_sample(
            correct=correct,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        status = "✓" if result["success"] else "✗"
        sentiment = result["data"].get("sentiment", "N/A") if result["data"] else "N/A"
        print(f"  [{status}] Review {i+1}: parse={'OK' if result['success'] else 'FAIL'}, sentiment={sentiment}")

        if result["success"] and result["data"]:
            # Show structured data preview
            data = result["data"]
            print(f"      confidence={data.get('confidence', 'N/A')}, recommend={data.get('recommendation', 'N/A')}")

        if not result["success"]:
            print(f"      Error: {result['error'][:80]}...")

    print(f"\n  Parse Success Rate: {experiment.accuracy:.1%}")
    print(f"  Avg Latency: {experiment.latency_ms:.0f}ms")
    print(f"  Total Cost: ${experiment.cost_usd:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Structured Outputs Demo")
    parser.add_argument(
        "--provider",
        choices=["anthropic"],
        default="anthropic",
        help="LLM provider (currently only anthropic supported for tool_use)",
    )
    args = parser.parse_args()

    # Initialize
    tracker = MetricsTracker("02_structured_outputs")

    try:
        client = AnthropicClient()
    except Exception as e:
        print(f"Error: Could not initialize Anthropic client: {e}")
        return 1

    print("\n" + "="*60)
    print("STRUCTURED OUTPUTS COMPARISON")
    print("Task: Extract structured review analysis")
    print(f"Reviews: {len(TEST_REVIEWS)}")
    print("="*60)

    # Run all three approaches
    run_raw_text_experiment(client, TEST_REVIEWS, tracker)
    run_json_mode_experiment(client, TEST_REVIEWS, tracker)
    run_tool_use_experiment(client, TEST_REVIEWS, tracker)

    # Print comparison
    print(tracker.compare(baseline_approach="raw_text"))

    # Save results
    tracker.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
