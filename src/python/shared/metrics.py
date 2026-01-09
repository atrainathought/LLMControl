"""
Metrics tracking and comparison for LLM experiments.

This module provides tools to:
1. Track performance metrics (accuracy, latency, cost, tokens)
2. Compare different approaches
3. Generate reports and visualizations
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics


@dataclass
class MetricsResult:
    """Single experiment result."""
    experiment_name: str
    approach: str  # e.g., "zero-shot", "few-shot", "chain-of-thought"
    provider: str
    model: str

    # Core metrics
    accuracy: Optional[float] = None  # 0-1 score if applicable
    correct_count: int = 0
    total_count: int = 0

    # Performance metrics
    latency_ms: float = 0.0
    latency_samples: List[float] = field(default_factory=list)

    # Cost metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: Dict[str, Any] = field(default_factory=dict)

    def add_sample(
        self,
        correct: bool = None,
        latency_ms: float = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ):
        """Add a single sample to the metrics."""
        if correct is not None:
            self.total_count += 1
            if correct:
                self.correct_count += 1
            self.accuracy = self.correct_count / self.total_count

        if latency_ms > 0:
            self.latency_samples.append(latency_ms)
            self.latency_ms = statistics.mean(self.latency_samples)

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.cost_usd += cost_usd

    def summary(self) -> Dict[str, Any]:
        """Get a summary dict for display."""
        latency_stats = {}
        if self.latency_samples:
            latency_stats = {
                "mean_ms": round(statistics.mean(self.latency_samples), 1),
                "min_ms": round(min(self.latency_samples), 1),
                "max_ms": round(max(self.latency_samples), 1),
            }
            if len(self.latency_samples) > 1:
                latency_stats["std_ms"] = round(statistics.stdev(self.latency_samples), 1)

        return {
            "experiment": self.experiment_name,
            "approach": self.approach,
            "provider": self.provider,
            "model": self.model,
            "accuracy": f"{self.accuracy:.1%}" if self.accuracy is not None else "N/A",
            "samples": self.total_count,
            "latency": latency_stats,
            "tokens": self.total_tokens,
            "cost_usd": f"${self.cost_usd:.6f}",
        }


class MetricsTracker:
    """Track and compare metrics across multiple experiments."""

    def __init__(self, module_name: str, save_dir: str = None):
        self.module_name = module_name
        self.results: List[MetricsResult] = []
        self.save_dir = Path(save_dir) if save_dir else Path("data/metrics")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        experiment_name: str,
        approach: str,
        provider: str,
        model: str,
    ) -> MetricsResult:
        """Create a new experiment result tracker."""
        result = MetricsResult(
            experiment_name=experiment_name,
            approach=approach,
            provider=provider,
            model=model,
        )
        self.results.append(result)
        return result

    def compare(self, baseline_approach: str = None) -> str:
        """Generate a comparison table of all results."""
        if not self.results:
            return "No results to compare."

        # Find baseline if specified
        baseline = None
        if baseline_approach:
            for r in self.results:
                if r.approach == baseline_approach:
                    baseline = r
                    break

        # Build comparison table
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"METRICS COMPARISON: {self.module_name}")
        lines.append(f"{'='*80}\n")

        # Header
        header = f"{'Approach':<20} {'Provider':<12} {'Accuracy':<10} {'Latency':<12} {'Tokens':<10} {'Cost':<12}"
        lines.append(header)
        lines.append("-" * 80)

        for r in self.results:
            acc = f"{r.accuracy:.1%}" if r.accuracy is not None else "N/A"
            lat = f"{r.latency_ms:.0f}ms" if r.latency_ms else "N/A"
            cost = f"${r.cost_usd:.6f}"

            # Calculate improvement vs baseline
            improvement = ""
            if baseline and r != baseline and r.accuracy and baseline.accuracy:
                diff = r.accuracy - baseline.accuracy
                improvement = f" ({'+' if diff > 0 else ''}{diff:.1%})"

            row = f"{r.approach:<20} {r.provider:<12} {acc + improvement:<10} {lat:<12} {r.total_tokens:<10} {cost:<12}"
            lines.append(row)

        lines.append("-" * 80)

        # Summary insights
        if len(self.results) > 1:
            best_acc = max((r for r in self.results if r.accuracy), key=lambda x: x.accuracy, default=None)
            lowest_cost = min(self.results, key=lambda x: x.cost_usd)
            fastest = min((r for r in self.results if r.latency_ms), key=lambda x: x.latency_ms, default=None)

            lines.append("\nINSIGHTS:")
            if best_acc:
                lines.append(f"  Best accuracy: {best_acc.approach} ({best_acc.accuracy:.1%})")
            if fastest:
                lines.append(f"  Fastest: {fastest.approach} ({fastest.latency_ms:.0f}ms)")
            lines.append(f"  Lowest cost: {lowest_cost.approach} (${lowest_cost.cost_usd:.6f})")

        return "\n".join(lines)

    def save(self, filename: str = None):
        """Save results to JSON file."""
        filename = filename or f"{self.module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.save_dir / filename

        data = {
            "module": self.module_name,
            "timestamp": datetime.now().isoformat(),
            "results": [asdict(r) for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Metrics saved to: {filepath}")
        return filepath

    def load(self, filepath: str):
        """Load results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        self.module_name = data["module"]
        self.results = [MetricsResult(**r) for r in data["results"]]
        return self


# Convenience function for timing
def timed(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms
    return wrapper


# Quick test
if __name__ == "__main__":
    print("Testing MetricsTracker...")

    tracker = MetricsTracker("test_module")

    # Simulate experiments
    exp1 = tracker.create_experiment("classification", "zero-shot", "openai", "gpt-4o-mini")
    exp1.add_sample(correct=True, latency_ms=150, input_tokens=100, output_tokens=50, cost_usd=0.0001)
    exp1.add_sample(correct=True, latency_ms=140, input_tokens=100, output_tokens=45, cost_usd=0.0001)
    exp1.add_sample(correct=False, latency_ms=160, input_tokens=100, output_tokens=55, cost_usd=0.0001)

    exp2 = tracker.create_experiment("classification", "few-shot", "openai", "gpt-4o-mini")
    exp2.add_sample(correct=True, latency_ms=180, input_tokens=300, output_tokens=50, cost_usd=0.0002)
    exp2.add_sample(correct=True, latency_ms=175, input_tokens=300, output_tokens=48, cost_usd=0.0002)
    exp2.add_sample(correct=True, latency_ms=185, input_tokens=300, output_tokens=52, cost_usd=0.0002)

    print(tracker.compare(baseline_approach="zero-shot"))
