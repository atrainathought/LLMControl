"""
Evals Framework Module.

Provides systematic evaluation of LLM outputs using:
- Code-based evaluators (deterministic)
- Semantic evaluators (embedding-based)
- LLM-as-judge evaluators (using Claude)

Usage:
    from 08_evals_framework import EvalPipeline, TestCase
    from 08_evals_framework.evaluators import LLMJudgeEvaluator

    pipeline = EvalPipeline(llm_client=client)
    pipeline.add_evaluator(LLMJudgeEvaluator(client))
    report = pipeline.run(test_cases)
"""

from .evaluators import (
    EvalType,
    EvalResult,
    BaseEvaluator,
    ExactMatchEvaluator,
    ContainsAllEvaluator,
    RegexMatchEvaluator,
    JSONSchemaEvaluator,
    LengthEvaluator,
    SemanticSimilarityEvaluator,
    LLMJudgeEvaluator,
    FaithfulnessEvaluator,
    CompositeEvaluator,
)

from .metrics import (
    MetricCategory,
    MetricResult,
    EvalMetrics,
    accuracy,
    precision,
    recall,
    f1_score,
    average_score,
    pass_rate,
)

from .pipeline import (
    TestCase,
    EvalRun,
    EvalReport,
    EvalPipeline,
    ModelComparison,
    quick_eval,
    eval_qa_dataset,
)

from .test_cases import (
    get_test_suite,
    create_custom_test_suite,
    QA_TEST_CASES,
    RAG_TEST_CASES,
    CLASSIFICATION_TEST_CASES,
)

__all__ = [
    # Evaluators
    "EvalType",
    "EvalResult",
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "ContainsAllEvaluator",
    "RegexMatchEvaluator",
    "JSONSchemaEvaluator",
    "LengthEvaluator",
    "SemanticSimilarityEvaluator",
    "LLMJudgeEvaluator",
    "FaithfulnessEvaluator",
    "CompositeEvaluator",
    # Metrics
    "MetricCategory",
    "MetricResult",
    "EvalMetrics",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "average_score",
    "pass_rate",
    # Pipeline
    "TestCase",
    "EvalRun",
    "EvalReport",
    "EvalPipeline",
    "ModelComparison",
    "quick_eval",
    "eval_qa_dataset",
    # Test Cases
    "get_test_suite",
    "create_custom_test_suite",
    "QA_TEST_CASES",
    "RAG_TEST_CASES",
    "CLASSIFICATION_TEST_CASES",
]
