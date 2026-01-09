"""
Evaluation Metrics Definitions.

Provides standardized metrics for evaluating LLM outputs:
- Accuracy metrics (for classification/factual tasks)
- Quality metrics (for open-ended generation)
- RAG-specific metrics (retrieval quality, faithfulness)
- Cost and efficiency metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import statistics


class MetricCategory(Enum):
    """Categories of evaluation metrics."""
    ACCURACY = "accuracy"
    QUALITY = "quality"
    RETRIEVAL = "retrieval"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    name: str
    value: float
    category: MetricCategory
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.value <= 1.0:
            return f"{self.name}: {self.value:.2%}"
        return f"{self.name}: {self.value:.2f}"


# =============================================================================
# ACCURACY METRICS
# =============================================================================

def accuracy(correct: int, total: int) -> MetricResult:
    """Simple accuracy: correct / total."""
    value = correct / total if total > 0 else 0
    return MetricResult(
        name="Accuracy",
        value=value,
        category=MetricCategory.ACCURACY,
        details={"correct": correct, "total": total}
    )


def precision(true_positives: int, false_positives: int) -> MetricResult:
    """Precision: TP / (TP + FP)."""
    total = true_positives + false_positives
    value = true_positives / total if total > 0 else 0
    return MetricResult(
        name="Precision",
        value=value,
        category=MetricCategory.ACCURACY,
        details={"true_positives": true_positives, "false_positives": false_positives}
    )


def recall(true_positives: int, false_negatives: int) -> MetricResult:
    """Recall: TP / (TP + FN)."""
    total = true_positives + false_negatives
    value = true_positives / total if total > 0 else 0
    return MetricResult(
        name="Recall",
        value=value,
        category=MetricCategory.ACCURACY,
        details={"true_positives": true_positives, "false_negatives": false_negatives}
    )


def f1_score(precision_val: float, recall_val: float) -> MetricResult:
    """F1 Score: 2 * (precision * recall) / (precision + recall)."""
    total = precision_val + recall_val
    value = 2 * (precision_val * recall_val) / total if total > 0 else 0
    return MetricResult(
        name="F1 Score",
        value=value,
        category=MetricCategory.ACCURACY,
        details={"precision": precision_val, "recall": recall_val}
    )


# =============================================================================
# QUALITY METRICS
# =============================================================================

def average_score(scores: List[float]) -> MetricResult:
    """Average of all scores."""
    value = statistics.mean(scores) if scores else 0
    return MetricResult(
        name="Average Score",
        value=value,
        category=MetricCategory.QUALITY,
        details={"count": len(scores), "min": min(scores) if scores else 0, "max": max(scores) if scores else 0}
    )


def pass_rate(passed: int, total: int, threshold: float = 0.7) -> MetricResult:
    """Percentage of evaluations that passed (score >= threshold)."""
    value = passed / total if total > 0 else 0
    return MetricResult(
        name="Pass Rate",
        value=value,
        category=MetricCategory.QUALITY,
        details={"passed": passed, "total": total, "threshold": threshold}
    )


def score_distribution(scores: List[float], bins: int = 5) -> MetricResult:
    """Distribution of scores across bins."""
    if not scores:
        return MetricResult(
            name="Score Distribution",
            value=0,
            category=MetricCategory.QUALITY,
            details={"bins": {}}
        )

    # Create bins
    bin_size = 1.0 / bins
    distribution = {f"{i*bin_size:.1f}-{(i+1)*bin_size:.1f}": 0 for i in range(bins)}

    for score in scores:
        bin_idx = min(int(score / bin_size), bins - 1)
        bin_key = f"{bin_idx*bin_size:.1f}-{(bin_idx+1)*bin_size:.1f}"
        distribution[bin_key] += 1

    return MetricResult(
        name="Score Distribution",
        value=statistics.mean(scores),
        category=MetricCategory.QUALITY,
        details={"bins": distribution, "count": len(scores)}
    )


def consistency(scores: List[float]) -> MetricResult:
    """Measure consistency via standard deviation (lower is more consistent)."""
    if len(scores) < 2:
        return MetricResult(
            name="Consistency",
            value=1.0,
            category=MetricCategory.QUALITY,
            details={"std_dev": 0}
        )

    std_dev = statistics.stdev(scores)
    # Convert to 0-1 scale where 1 is most consistent
    value = max(0, 1 - std_dev)

    return MetricResult(
        name="Consistency",
        value=value,
        category=MetricCategory.QUALITY,
        details={"std_dev": std_dev, "variance": statistics.variance(scores)}
    )


# =============================================================================
# RETRIEVAL METRICS (for RAG)
# =============================================================================

def retrieval_precision(relevant_retrieved: int, total_retrieved: int) -> MetricResult:
    """Precision of retrieved documents."""
    value = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
    return MetricResult(
        name="Retrieval Precision",
        value=value,
        category=MetricCategory.RETRIEVAL,
        details={"relevant": relevant_retrieved, "retrieved": total_retrieved}
    )


def retrieval_recall(relevant_retrieved: int, total_relevant: int) -> MetricResult:
    """Recall of retrieved documents."""
    value = relevant_retrieved / total_relevant if total_relevant > 0 else 0
    return MetricResult(
        name="Retrieval Recall",
        value=value,
        category=MetricCategory.RETRIEVAL,
        details={"relevant_retrieved": relevant_retrieved, "total_relevant": total_relevant}
    )


def mean_reciprocal_rank(ranks: List[int]) -> MetricResult:
    """
    Mean Reciprocal Rank (MRR).

    Args:
        ranks: List of ranks where the first relevant document appeared (1-indexed)
               Use 0 or None for queries with no relevant results
    """
    if not ranks:
        return MetricResult(
            name="MRR",
            value=0,
            category=MetricCategory.RETRIEVAL,
            details={"queries": 0}
        )

    reciprocal_ranks = [1/r if r and r > 0 else 0 for r in ranks]
    value = statistics.mean(reciprocal_ranks)

    return MetricResult(
        name="MRR",
        value=value,
        category=MetricCategory.RETRIEVAL,
        details={"queries": len(ranks), "reciprocal_ranks": reciprocal_ranks}
    )


def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> MetricResult:
    """
    Normalized Discounted Cumulative Gain at K.

    Args:
        relevance_scores: Relevance scores of retrieved docs (in retrieval order)
        k: Number of top results to consider
    """
    import math

    def dcg(scores: List[float]) -> float:
        return sum(score / math.log2(i + 2) for i, score in enumerate(scores[:k]))

    actual_dcg = dcg(relevance_scores)
    ideal_dcg = dcg(sorted(relevance_scores, reverse=True))

    value = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0

    return MetricResult(
        name=f"NDCG@{k}",
        value=value,
        category=MetricCategory.RETRIEVAL,
        details={"dcg": actual_dcg, "ideal_dcg": ideal_dcg, "k": k}
    )


def context_relevance(relevance_scores: List[float]) -> MetricResult:
    """Average relevance of retrieved context."""
    value = statistics.mean(relevance_scores) if relevance_scores else 0
    return MetricResult(
        name="Context Relevance",
        value=value,
        category=MetricCategory.RETRIEVAL,
        details={"count": len(relevance_scores), "scores": relevance_scores}
    )


def faithfulness_score(scores: List[float]) -> MetricResult:
    """Average faithfulness (groundedness) of responses."""
    value = statistics.mean(scores) if scores else 0
    return MetricResult(
        name="Faithfulness",
        value=value,
        category=MetricCategory.RETRIEVAL,
        details={"count": len(scores), "min": min(scores) if scores else 0}
    )


# =============================================================================
# EFFICIENCY METRICS
# =============================================================================

def average_latency(latencies_ms: List[float]) -> MetricResult:
    """Average latency in milliseconds."""
    value = statistics.mean(latencies_ms) if latencies_ms else 0
    return MetricResult(
        name="Avg Latency",
        value=value,
        category=MetricCategory.EFFICIENCY,
        details={
            "count": len(latencies_ms),
            "min_ms": min(latencies_ms) if latencies_ms else 0,
            "max_ms": max(latencies_ms) if latencies_ms else 0,
            "p50_ms": statistics.median(latencies_ms) if latencies_ms else 0,
        }
    )


def total_cost(costs_usd: List[float]) -> MetricResult:
    """Total cost in USD."""
    value = sum(costs_usd)
    return MetricResult(
        name="Total Cost",
        value=value,
        category=MetricCategory.EFFICIENCY,
        details={"count": len(costs_usd), "avg_cost": statistics.mean(costs_usd) if costs_usd else 0}
    )


def cost_per_eval(total_cost_usd: float, num_evals: int) -> MetricResult:
    """Cost per evaluation."""
    value = total_cost_usd / num_evals if num_evals > 0 else 0
    return MetricResult(
        name="Cost Per Eval",
        value=value,
        category=MetricCategory.EFFICIENCY,
        details={"total_cost": total_cost_usd, "num_evals": num_evals}
    )


def tokens_per_eval(total_tokens: int, num_evals: int) -> MetricResult:
    """Average tokens per evaluation."""
    value = total_tokens / num_evals if num_evals > 0 else 0
    return MetricResult(
        name="Tokens Per Eval",
        value=value,
        category=MetricCategory.EFFICIENCY,
        details={"total_tokens": total_tokens, "num_evals": num_evals}
    )


# =============================================================================
# SAFETY METRICS
# =============================================================================

def toxicity_rate(toxic_count: int, total: int) -> MetricResult:
    """Rate of toxic outputs."""
    value = toxic_count / total if total > 0 else 0
    return MetricResult(
        name="Toxicity Rate",
        value=value,
        category=MetricCategory.SAFETY,
        details={"toxic": toxic_count, "total": total}
    )


def refusal_rate(refused: int, total: int) -> MetricResult:
    """Rate of appropriate refusals (for unsafe prompts)."""
    value = refused / total if total > 0 else 0
    return MetricResult(
        name="Refusal Rate",
        value=value,
        category=MetricCategory.SAFETY,
        details={"refused": refused, "total": total}
    )


# =============================================================================
# METRICS AGGREGATOR
# =============================================================================

@dataclass
class EvalMetrics:
    """Aggregated metrics from an evaluation run."""

    total_samples: int = 0
    passed: int = 0
    failed: int = 0

    scores: List[float] = field(default_factory=list)
    latencies_ms: List[float] = field(default_factory=list)
    costs_usd: List[float] = field(default_factory=list)
    tokens_used: List[int] = field(default_factory=list)

    custom_metrics: Dict[str, MetricResult] = field(default_factory=dict)

    def add_result(
        self,
        passed: bool,
        score: float,
        latency_ms: float = 0,
        cost_usd: float = 0,
        tokens: int = 0
    ):
        """Add a single evaluation result."""
        self.total_samples += 1
        if passed:
            self.passed += 1
        else:
            self.failed += 1

        self.scores.append(score)
        if latency_ms > 0:
            self.latencies_ms.append(latency_ms)
        if cost_usd > 0:
            self.costs_usd.append(cost_usd)
        if tokens > 0:
            self.tokens_used.append(tokens)

    def add_custom_metric(self, metric: MetricResult):
        """Add a custom metric."""
        self.custom_metrics[metric.name] = metric

    def get_summary(self) -> Dict[str, MetricResult]:
        """Get all computed metrics."""
        metrics = {
            "accuracy": accuracy(self.passed, self.total_samples),
            "pass_rate": pass_rate(self.passed, self.total_samples),
            "avg_score": average_score(self.scores),
            "consistency": consistency(self.scores),
            "score_distribution": score_distribution(self.scores),
        }

        if self.latencies_ms:
            metrics["avg_latency"] = average_latency(self.latencies_ms)

        if self.costs_usd:
            metrics["total_cost"] = total_cost(self.costs_usd)
            metrics["cost_per_eval"] = cost_per_eval(sum(self.costs_usd), self.total_samples)

        if self.tokens_used:
            metrics["tokens_per_eval"] = tokens_per_eval(sum(self.tokens_used), self.total_samples)

        # Add custom metrics
        metrics.update(self.custom_metrics)

        return metrics

    def print_summary(self, title: str = "Evaluation Results"):
        """Print a formatted summary of metrics."""
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")

        metrics = self.get_summary()

        # Group by category
        by_category: Dict[MetricCategory, List[MetricResult]] = {}
        for metric in metrics.values():
            category = metric.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(metric)

        for category in MetricCategory:
            if category in by_category:
                print(f"\n{category.value.upper()}:")
                for metric in by_category[category]:
                    print(f"  {metric}")

        print(f"\n{'='*60}")
