#!/usr/bin/env python3
"""
LoRA Fine-tuning Demo

This script demonstrates:
1. Training a LoRA adapter on customer service data
2. Comparing base model vs fine-tuned model outputs
3. Measuring improvement in tone and professionalism

Usage:
    # Train the model (takes ~5-10 minutes on GPU)
    python demo.py --train

    # Test with a pre-trained model
    python demo.py --test

    # Full demo (train + test)
    python demo.py --full
"""

import sys
import argparse
import time
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.metrics import MetricsTracker

from dataset import TEST_DATA, format_for_inference, get_dataset_stats
from trainer import LoRATrainer, LoRAConfig, run_training, load_trained_model


def check_gpu():
    """Check GPU availability and memory."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_free = (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated(0)) / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"Memory: {memory_free:.1f}GB free / {memory_total:.1f}GB total")
        return True
    else:
        print("No GPU available. Training will be slow on CPU.")
        return False


def evaluate_response(response: str, test_case: dict) -> dict:
    """Evaluate a response against expected criteria."""
    response_lower = response.lower()

    # Check for expected professional tone indicators
    tone_hits = sum(1 for word in test_case["expected_tone"]
                    if word.lower() in response_lower)
    tone_score = tone_hits / len(test_case["expected_tone"])

    # Check that informal words are NOT present
    informal_present = any(word.lower() in response_lower
                          for word in test_case["should_not_contain"])

    # Length check (professional responses should be substantial)
    is_substantial = len(response) > 50

    # Overall quality
    quality_score = (
        (tone_score * 0.5) +
        (0.3 if not informal_present else 0) +
        (0.2 if is_substantial else 0)
    )

    return {
        "tone_score": tone_score,
        "no_informal": not informal_present,
        "substantial": is_substantial,
        "quality_score": quality_score,
        "response_length": len(response),
    }


def run_comparison(base_trainer: LoRATrainer, finetuned_trainer: LoRATrainer):
    """Compare base model vs fine-tuned model on test cases."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Base vs Fine-tuned")
    print("=" * 70)

    base_scores = []
    finetuned_scores = []

    for i, test in enumerate(TEST_DATA):
        print(f"\n--- Test {i + 1} ---")
        print(f"Input: {test['input']}")

        # Generate with base model
        base_response = base_trainer.generate(test['input'])
        base_eval = evaluate_response(base_response, test)
        base_scores.append(base_eval['quality_score'])

        # Generate with fine-tuned model
        ft_response = finetuned_trainer.generate(test['input'])
        ft_eval = evaluate_response(ft_response, test)
        finetuned_scores.append(ft_eval['quality_score'])

        print(f"\nBase Model:")
        print(f"  Response: {base_response[:150]}...")
        print(f"  Tone Score: {base_eval['tone_score']:.0%}")
        print(f"  No Informal: {base_eval['no_informal']}")
        print(f"  Quality: {base_eval['quality_score']:.0%}")

        print(f"\nFine-tuned Model:")
        print(f"  Response: {ft_response[:150]}...")
        print(f"  Tone Score: {ft_eval['tone_score']:.0%}")
        print(f"  No Informal: {ft_eval['no_informal']}")
        print(f"  Quality: {ft_eval['quality_score']:.0%}")

    # Summary
    avg_base = sum(base_scores) / len(base_scores)
    avg_ft = sum(finetuned_scores) / len(finetuned_scores)
    improvement = (avg_ft - avg_base) / avg_base * 100 if avg_base > 0 else 0

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nBase Model Average Quality: {avg_base:.0%}")
    print(f"Fine-tuned Average Quality: {avg_ft:.0%}")
    print(f"Improvement: {improvement:+.1f}%")

    return {
        "base_avg": avg_base,
        "finetuned_avg": avg_ft,
        "improvement_pct": improvement,
        "base_scores": base_scores,
        "finetuned_scores": finetuned_scores,
    }


def demo_training(output_dir: str):
    """Run the training demo."""
    print("\n" + "=" * 70)
    print("LORA FINE-TUNING DEMO")
    print("=" * 70)

    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("\nWarning: Training without GPU will be very slow.")
        print("Consider using Google Colab or a cloud GPU instance.")

    # Dataset stats
    stats = get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"  Training examples: {stats['train_examples']}")
    print(f"  Validation examples: {stats['val_examples']}")
    print(f"  Test examples: {stats['test_examples']}")
    print(f"  Avg input length: {stats['avg_input_chars']:.0f} chars")
    print(f"  Avg output length: {stats['avg_output_chars']:.0f} chars")

    # Train
    print("\n" + "-" * 70)
    print("TRAINING")
    print("-" * 70)

    start_time = time.time()
    trainer = run_training(output_dir)
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time / 60:.1f} minutes")

    return trainer


def demo_inference(model_path: str):
    """Run inference demo with trained model."""
    print("\n" + "=" * 70)
    print("INFERENCE DEMO")
    print("=" * 70)

    # Load trained model
    trainer = load_trained_model(model_path)

    # Test on each test case
    print("\nGenerating responses...")
    for i, test in enumerate(TEST_DATA):
        print(f"\n--- Test {i + 1} ---")
        print(f"Input: {test['input']}")

        response = trainer.generate(test['input'])
        eval_result = evaluate_response(response, test)

        print(f"Response: {response}")
        print(f"Quality Score: {eval_result['quality_score']:.0%}")

    return trainer


def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning Demo")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--full", action="store_true", help="Full demo (train + test)")
    parser.add_argument(
        "--model-path",
        default="./lora_output",
        help="Path to save/load model"
    )
    args = parser.parse_args()

    # Default to full demo if no args
    if not args.train and not args.test and not args.full:
        args.full = True

    tracker = MetricsTracker("07_lora_finetuning")

    try:
        if args.train or args.full:
            # Training
            trainer = demo_training(args.model_path)

            if args.full:
                # Also run comparison
                print("\n" + "=" * 70)
                print("LOADING BASE MODEL FOR COMPARISON")
                print("=" * 70)

                # Load base model (without LoRA)
                base_config = LoRAConfig()
                base_trainer = LoRATrainer(base_config)
                base_trainer.load_model()

                # Compare
                results = run_comparison(base_trainer, trainer)

                # Track metrics
                exp_base = tracker.create_experiment(
                    experiment_name="tone_quality",
                    approach="base_model",
                    provider="local",
                    model="TinyLlama-1.1B",
                )
                for score in results["base_scores"]:
                    exp_base.add_sample(correct=score > 0.5, latency_ms=0)

                exp_ft = tracker.create_experiment(
                    experiment_name="tone_quality",
                    approach="lora_finetuned",
                    provider="local",
                    model="TinyLlama-1.1B-LoRA",
                )
                for score in results["finetuned_scores"]:
                    exp_ft.add_sample(correct=score > 0.5, latency_ms=0)

        elif args.test:
            # Just test
            if not Path(args.model_path).exists():
                print(f"Error: No trained model found at {args.model_path}")
                print("Run with --train first to train a model.")
                return 1

            trainer = demo_inference(args.model_path)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 70)
            print("OUT OF MEMORY ERROR")
            print("=" * 70)
            print("""
Your GPU doesn't have enough memory for this model configuration.

Options:
1. Reduce batch_size to 1 in trainer.py
2. Use a smaller model (e.g., distilgpt2)
3. Use Google Colab with a free T4 GPU
4. Use CPU (very slow): set device_map='cpu' in trainer.py
""")
            return 1
        raise

    # Print insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
LORA FINE-TUNING:

1. WHAT IS LORA?
   - Low-Rank Adaptation of Large Language Models
   - Freezes pretrained weights, adds small trainable matrices
   - Reduces trainable parameters from billions to millions

2. WHY USE LORA?
   - Memory efficient: Fine-tune on consumer GPUs (4GB VRAM)
   - Fast training: Minutes instead of hours
   - Portable: LoRA adapters are small (~10-50MB)
   - Composable: Can merge/switch adapters for different tasks

3. HYPERPARAMETERS
   - r (rank): Higher = more capacity, more memory. 8-64 typical
   - alpha: Scaling factor, usually 2x rank
   - target_modules: Which layers to adapt (q_proj, v_proj, etc.)

4. TRAINING DATA
   - Quality > Quantity: 20-100 good examples often sufficient
   - Format consistency: Use same prompt template
   - Task-specific: Narrow focus works better than general

5. LIMITATIONS
   - Won't add new knowledge (only adapts behavior)
   - Small models have capability ceiling
   - Still needs task-appropriate base model
""")

    print(tracker.compare())
    tracker.save()

    return 0


if __name__ == "__main__":
    sys.exit(main())
