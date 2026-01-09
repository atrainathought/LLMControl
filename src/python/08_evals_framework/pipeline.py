"""
Evaluation Pipeline.

Orchestrates running evaluations across test cases with multiple evaluators.
Supports:
- Running single or multiple evaluators on test suites
- Parallel evaluation execution
- Model comparison
- Generating evaluation reports
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from .evaluators import BaseEvaluator, EvalResult, EvalType
    from .metrics import EvalMetrics, MetricResult
except ImportError:
    from evaluators import BaseEvaluator, EvalResult, EvalType
    from metrics import EvalMetrics, MetricResult


try:
    from .test_cases import TestCase
except ImportError:
    from test_cases import TestCase


@dataclass
class EvalRun:
    """Results from a single evaluation run."""
    test_case: TestCase
    actual_output: str
    results: List[EvalResult]  # Results from each evaluator
    latency_ms: float = 0
    tokens_used: int = 0
    cost_usd: float = 0

    @property
    def passed(self) -> bool:
        """All evaluators must pass."""
        return all(r.passed for r in self.results)

    @property
    def average_score(self) -> float:
        """Average score across all evaluators."""
        if not self.results:
            return 0
        return sum(r.score for r in self.results) / len(self.results)


@dataclass
class EvalReport:
    """Complete evaluation report."""
    runs: List[EvalRun]
    metrics: EvalMetrics
    evaluators_used: List[str]
    total_time_ms: float

    def get_failures(self) -> List[EvalRun]:
        """Get all failed runs."""
        return [run for run in self.runs if not run.passed]

    def get_by_evaluator(self, evaluator_name: str) -> List[EvalResult]:
        """Get results filtered by evaluator."""
        results = []
        for run in self.runs:
            for result in run.results:
                if result.evaluator_name == evaluator_name:
                    results.append(result)
        return results

    def print_summary(self):
        """Print a summary of the evaluation report."""
        self.metrics.print_summary("Evaluation Pipeline Report")

        print(f"\nEvaluators: {', '.join(self.evaluators_used)}")
        print(f"Total Time: {self.total_time_ms:.0f}ms")

        failures = self.get_failures()
        if failures:
            print(f"\n--- Failed Cases ({len(failures)}) ---")
            for run in failures[:5]:  # Show first 5 failures
                print(f"\nTest: {run.test_case.id}")
                print(f"Input: {run.test_case.input[:100]}...")
                for result in run.results:
                    if not result.passed:
                        print(f"  [{result.evaluator_name}] Score: {result.score:.2%} - {result.reasoning}")


class EvalPipeline:
    """
    Evaluation Pipeline.

    Runs test cases through an LLM and evaluates outputs.
    """

    def __init__(
        self,
        llm_client = None,
        system_prompt: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Initialize the pipeline.

        Args:
            llm_client: LLM client to generate responses (optional if providing pre-computed outputs)
            system_prompt: System prompt for the LLM
            temperature: Temperature for generation
            max_tokens: Max tokens for generation
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.evaluators: List[BaseEvaluator] = []

    def add_evaluator(self, evaluator: BaseEvaluator) -> "EvalPipeline":
        """Add an evaluator to the pipeline."""
        self.evaluators.append(evaluator)
        return self

    def run(
        self,
        test_cases: List[TestCase],
        pre_computed_outputs: Dict[str, str] = None,
        parallel: bool = False,
        max_workers: int = 4,
    ) -> EvalReport:
        """
        Run the evaluation pipeline.

        Args:
            test_cases: List of test cases to evaluate
            pre_computed_outputs: Optional dict mapping test_case.id to output string
            parallel: Whether to run evaluations in parallel
            max_workers: Number of parallel workers

        Returns:
            EvalReport with all results
        """
        if not self.evaluators:
            raise ValueError("No evaluators added to pipeline")

        start_time = time.perf_counter()
        runs: List[EvalRun] = []
        metrics = EvalMetrics()

        if parallel and len(test_cases) > 1:
            runs = self._run_parallel(test_cases, pre_computed_outputs, max_workers)
        else:
            for test_case in test_cases:
                run = self._evaluate_single(test_case, pre_computed_outputs)
                runs.append(run)

        # Aggregate metrics
        for run in runs:
            metrics.add_result(
                passed=run.passed,
                score=run.average_score,
                latency_ms=run.latency_ms,
                cost_usd=run.cost_usd,
                tokens=run.tokens_used,
            )

        total_time = (time.perf_counter() - start_time) * 1000

        return EvalReport(
            runs=runs,
            metrics=metrics,
            evaluators_used=[e.name for e in self.evaluators],
            total_time_ms=total_time,
        )

    def _evaluate_single(
        self,
        test_case: TestCase,
        pre_computed_outputs: Dict[str, str] = None,
    ) -> EvalRun:
        """Evaluate a single test case."""
        # Get or generate output
        if pre_computed_outputs and test_case.id in pre_computed_outputs:
            actual_output = pre_computed_outputs[test_case.id]
            latency_ms = 0
            tokens_used = 0
            cost_usd = 0
        elif self.llm_client:
            response = self.llm_client.complete(
                prompt=test_case.input,
                system=self.system_prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            actual_output = response.content
            latency_ms = response.latency_ms
            tokens_used = response.total_tokens
            cost_usd = response.cost_usd
        else:
            raise ValueError(f"No output for test case {test_case.id} and no LLM client provided")

        # Run all evaluators
        results = []
        for evaluator in self.evaluators:
            result = evaluator.evaluate(
                actual=actual_output,
                expected=test_case.expected,
                context=test_case.context,
                question=test_case.input,
            )
            results.append(result)

        return EvalRun(
            test_case=test_case,
            actual_output=actual_output,
            results=results,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )

    def _run_parallel(
        self,
        test_cases: List[TestCase],
        pre_computed_outputs: Dict[str, str] = None,
        max_workers: int = 4,
    ) -> List[EvalRun]:
        """Run evaluations in parallel."""
        runs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._evaluate_single, tc, pre_computed_outputs): tc
                for tc in test_cases
            }

            for future in as_completed(futures):
                try:
                    run = future.result()
                    runs.append(run)
                except Exception as e:
                    test_case = futures[future]
                    print(f"Error evaluating {test_case.id}: {e}")

        return runs


class ModelComparison:
    """
    Compare multiple models on the same test suite.
    """

    def __init__(self, test_cases: List[TestCase], evaluators: List[BaseEvaluator]):
        self.test_cases = test_cases
        self.evaluators = evaluators
        self.results: Dict[str, EvalReport] = {}

    def add_model(
        self,
        name: str,
        llm_client,
        system_prompt: str = None,
    ) -> "ModelComparison":
        """Add a model to compare."""
        pipeline = EvalPipeline(
            llm_client=llm_client,
            system_prompt=system_prompt,
        )
        for evaluator in self.evaluators:
            pipeline.add_evaluator(evaluator)

        print(f"\nEvaluating model: {name}")
        report = pipeline.run(self.test_cases)
        self.results[name] = report

        return self

    def get_comparison(self) -> Dict[str, Dict[str, float]]:
        """Get comparison metrics across models."""
        comparison = {}

        for model_name, report in self.results.items():
            metrics = report.metrics.get_summary()
            comparison[model_name] = {
                metric_name: metric.value
                for metric_name, metric in metrics.items()
            }

        return comparison

    def print_comparison(self):
        """Print a comparison table."""
        comparison = self.get_comparison()

        if not comparison:
            print("No models evaluated yet")
            return

        # Get all metric names
        metric_names = list(next(iter(comparison.values())).keys())

        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)

        # Header
        header = "Metric".ljust(25) + "".join(m.ljust(20) for m in comparison.keys())
        print(header)
        print("-"*80)

        # Rows
        for metric in metric_names:
            row = metric.ljust(25)
            for model in comparison.keys():
                value = comparison[model].get(metric, 0)
                if value <= 1.0:
                    row += f"{value:.2%}".ljust(20)
                else:
                    row += f"{value:.2f}".ljust(20)
            print(row)

        print("="*80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_eval(
    actual: str,
    expected: str = None,
    evaluators: List[BaseEvaluator] = None,
) -> Dict[str, EvalResult]:
    """
    Quick evaluation of a single output.

    Args:
        actual: The output to evaluate
        expected: Expected output (optional)
        evaluators: List of evaluators to use

    Returns:
        Dict mapping evaluator name to result
    """
    from .evaluators import (
        ExactMatchEvaluator,
        ContainsAllEvaluator,
        LengthEvaluator,
    )

    if evaluators is None:
        evaluators = [
            LengthEvaluator(min_length=10, max_length=5000),
        ]
        if expected:
            evaluators.append(ExactMatchEvaluator(case_sensitive=False))

    results = {}
    for evaluator in evaluators:
        result = evaluator.evaluate(actual, expected)
        results[evaluator.name] = result

    return results


def eval_qa_dataset(
    llm_client,
    questions: List[str],
    expected_answers: List[str],
    system_prompt: str = None,
) -> EvalReport:
    """
    Evaluate an LLM on a Q&A dataset.

    Args:
        llm_client: LLM client to use
        questions: List of questions
        expected_answers: List of expected answers
        system_prompt: Optional system prompt

    Returns:
        EvalReport with results
    """
    from .evaluators import SemanticSimilarityEvaluator, ContainsAllEvaluator

    test_cases = [
        TestCase(
            id=f"q{i}",
            input=q,
            expected=a,
        )
        for i, (q, a) in enumerate(zip(questions, expected_answers))
    ]

    pipeline = EvalPipeline(
        llm_client=llm_client,
        system_prompt=system_prompt,
    )
    pipeline.add_evaluator(SemanticSimilarityEvaluator(threshold=0.7))

    return pipeline.run(test_cases)
