#!/usr/bin/env python3
"""
Prompt Engineering Demo

This script demonstrates and compares three prompting techniques:
1. Zero-shot prompting
2. Few-shot prompting
3. Chain-of-thought prompting

Run this script to see how each approach performs on a sentiment classification task.

Usage:
    python demo.py                    # Run with OpenAI (default)
    python demo.py --provider anthropic   # Run with Anthropic
    python demo.py --provider both        # Compare both providers
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm_client import OpenAIClient, AnthropicClient, LLMClient
from shared.metrics import MetricsTracker

from prompts import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    CHAIN_OF_THOUGHT_PROMPT,
    CHAIN_OF_THOUGHT_IMPROVED_PROMPT,
    SYSTEM_PROMPTS,
    TEST_REVIEWS,
    format_prompt,
    extract_sentiment,
)


def run_experiment(
    client: LLMClient,
    approach: str,
    prompt_template: str,
    tracker: MetricsTracker,
) -> None:
    """Run an experiment with a specific prompting approach."""

    experiment = tracker.create_experiment(
        experiment_name="sentiment_classification",
        approach=approach,
        provider=client.get_name().split("/")[0].lower(),
        model=client.get_name().split("/")[1],
    )

    system_prompt = SYSTEM_PROMPTS.get(approach, "")

    print(f"\n{'='*60}")
    print(f"Running {approach.upper()} with {client.get_name()}")
    print(f"{'='*60}")

    for i, item in enumerate(TEST_REVIEWS):
        review = item["review"]
        expected = item["label"]

        # Format and send prompt
        prompt = format_prompt(prompt_template, review)
        response = client.complete(prompt, system=system_prompt, temperature=0.0)

        # Extract prediction
        predicted = extract_sentiment(response.content, approach)
        correct = predicted == expected

        # Log metrics
        experiment.add_sample(
            correct=correct,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
        )

        # Print result
        status = "✓" if correct else "✗"
        print(f"  [{status}] Review {i+1}: {predicted} (expected: {expected})")

        if approach.startswith("chain-of-thought") and not correct:
            # Show reasoning for incorrect chain-of-thought responses
            print(f"      Response preview: {response.content[:100]}...")

    print(f"\n  Accuracy: {experiment.accuracy:.1%} ({experiment.correct_count}/{experiment.total_count})")
    print(f"  Avg Latency: {experiment.latency_ms:.0f}ms")
    print(f"  Total Cost: ${experiment.cost_usd:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Prompt Engineering Demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "both"],
        default="openai",
        help="LLM provider to use",
    )
    args = parser.parse_args()

    # Initialize metrics tracker
    tracker = MetricsTracker("01_prompt_engineering")

    # Define approaches (4 techniques to compare)
    approaches = [
        ("zero-shot", ZERO_SHOT_PROMPT),
        ("few-shot", FEW_SHOT_PROMPT),
        ("chain-of-thought", CHAIN_OF_THOUGHT_PROMPT),
        ("chain-of-thought-improved", CHAIN_OF_THOUGHT_IMPROVED_PROMPT),
    ]

    # Get clients based on provider selection
    clients = []
    if args.provider in ("openai", "both"):
        try:
            clients.append(OpenAIClient())
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")

    if args.provider in ("anthropic", "both"):
        try:
            clients.append(AnthropicClient())
        except Exception as e:
            print(f"Warning: Could not initialize Anthropic client: {e}")

    if not clients:
        print("Error: No LLM clients available. Check your API keys in .env")
        return 1

    # Run experiments
    print("\n" + "="*60)
    print("PROMPT ENGINEERING COMPARISON")
    print("Task: Sentiment Classification (10 product reviews)")
    print("="*60)

    for client in clients:
        for approach, prompt_template in approaches:
            try:
                run_experiment(client, approach, prompt_template, tracker)
            except Exception as e:
                print(f"Error in {approach} with {client.get_name()}: {e}")

    # Print comparison
    print(tracker.compare(baseline_approach="zero-shot"))

    # Save results
    tracker.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
