"""
Evaluators for LLM Output Assessment.

Three types of evaluators:
1. CodeBasedEvaluator - Deterministic checks (exact match, regex, JSON schema)
2. SemanticEvaluator - Embedding-based similarity scoring
3. LLMJudgeEvaluator - Uses an LLM to evaluate output quality

Each evaluator returns a standardized EvalResult.
"""

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class EvalType(Enum):
    """Types of evaluation methods."""
    CODE_BASED = "code_based"
    SEMANTIC = "semantic"
    LLM_JUDGE = "llm_judge"


@dataclass
class EvalResult:
    """Standardized evaluation result."""
    passed: bool
    score: float  # 0.0 to 1.0
    eval_type: EvalType
    evaluator_name: str
    details: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def evaluate(self, actual: str, expected: Any = None, **kwargs) -> EvalResult:
        """Evaluate the actual output against expected criteria."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the evaluator name."""
        pass


# =============================================================================
# CODE-BASED EVALUATORS
# =============================================================================

class ExactMatchEvaluator(BaseEvaluator):
    """Check if output exactly matches expected value."""

    def __init__(self, case_sensitive: bool = True, strip_whitespace: bool = True):
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    @property
    def name(self) -> str:
        return "ExactMatch"

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        actual_clean = actual.strip() if self.strip_whitespace else actual
        expected_clean = expected.strip() if self.strip_whitespace else expected

        if not self.case_sensitive:
            actual_clean = actual_clean.lower()
            expected_clean = expected_clean.lower()

        passed = actual_clean == expected_clean

        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            eval_type=EvalType.CODE_BASED,
            evaluator_name=self.name,
            details={
                "actual": actual_clean[:100],
                "expected": expected_clean[:100],
                "case_sensitive": self.case_sensitive,
            },
            reasoning="Exact match" if passed else "Values do not match exactly"
        )


class ContainsAllEvaluator(BaseEvaluator):
    """Check if output contains all required keywords."""

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "ContainsAll"

    def evaluate(self, actual: str, expected: List[str], **kwargs) -> EvalResult:
        search_text = actual if self.case_sensitive else actual.lower()

        found = []
        missing = []

        for keyword in expected:
            search_keyword = keyword if self.case_sensitive else keyword.lower()
            if search_keyword in search_text:
                found.append(keyword)
            else:
                missing.append(keyword)

        score = len(found) / len(expected) if expected else 1.0
        passed = len(missing) == 0

        return EvalResult(
            passed=passed,
            score=score,
            eval_type=EvalType.CODE_BASED,
            evaluator_name=self.name,
            details={
                "found": found,
                "missing": missing,
                "total_keywords": len(expected),
            },
            reasoning=f"Found {len(found)}/{len(expected)} keywords" +
                     (f". Missing: {missing}" if missing else "")
        )


class RegexMatchEvaluator(BaseEvaluator):
    """Check if output matches a regex pattern."""

    @property
    def name(self) -> str:
        return "RegexMatch"

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        """Expected is a regex pattern string."""
        try:
            pattern = re.compile(expected, re.IGNORECASE | re.DOTALL)
            match = pattern.search(actual)
            passed = match is not None

            return EvalResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                eval_type=EvalType.CODE_BASED,
                evaluator_name=self.name,
                details={
                    "pattern": expected,
                    "matched": match.group(0)[:100] if match else None,
                },
                reasoning="Pattern matched" if passed else "Pattern not found"
            )
        except re.error as e:
            return EvalResult(
                passed=False,
                score=0.0,
                eval_type=EvalType.CODE_BASED,
                evaluator_name=self.name,
                details={"error": str(e)},
                reasoning=f"Invalid regex pattern: {e}"
            )


class JSONSchemaEvaluator(BaseEvaluator):
    """Validate that output is valid JSON matching a schema."""

    @property
    def name(self) -> str:
        return "JSONSchema"

    def evaluate(self, actual: str, expected: Dict[str, Any] = None, **kwargs) -> EvalResult:
        """
        Expected is a simple schema dict with field names and types.
        Example: {"name": str, "age": int, "active": bool}
        """
        # Try to parse JSON
        try:
            # Extract JSON from response if wrapped in markdown
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', actual)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', actual)
                json_str = json_match.group(0) if json_match else actual

            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            return EvalResult(
                passed=False,
                score=0.0,
                eval_type=EvalType.CODE_BASED,
                evaluator_name=self.name,
                details={"parse_error": str(e)},
                reasoning=f"Invalid JSON: {e}"
            )

        # If no schema provided, just check if valid JSON
        if expected is None:
            return EvalResult(
                passed=True,
                score=1.0,
                eval_type=EvalType.CODE_BASED,
                evaluator_name=self.name,
                details={"parsed_type": type(parsed).__name__},
                reasoning="Valid JSON"
            )

        # Validate against schema
        errors = []
        matched_fields = 0
        total_fields = len(expected)

        for field_name, field_type in expected.items():
            if field_name not in parsed:
                errors.append(f"Missing field: {field_name}")
            elif not isinstance(parsed[field_name], field_type):
                errors.append(f"Field '{field_name}' should be {field_type.__name__}, got {type(parsed[field_name]).__name__}")
            else:
                matched_fields += 1

        score = matched_fields / total_fields if total_fields > 0 else 1.0
        passed = len(errors) == 0

        return EvalResult(
            passed=passed,
            score=score,
            eval_type=EvalType.CODE_BASED,
            evaluator_name=self.name,
            details={
                "schema": {k: v.__name__ for k, v in expected.items()},
                "errors": errors,
                "matched_fields": matched_fields,
            },
            reasoning="Schema valid" if passed else f"Schema errors: {'; '.join(errors)}"
        )


class LengthEvaluator(BaseEvaluator):
    """Check if output length is within bounds."""

    def __init__(self, min_length: int = 0, max_length: int = float('inf')):
        self.min_length = min_length
        self.max_length = max_length

    @property
    def name(self) -> str:
        return "Length"

    def evaluate(self, actual: str, expected: Any = None, **kwargs) -> EvalResult:
        length = len(actual)
        within_bounds = self.min_length <= length <= self.max_length

        # Score based on how well it fits the target range
        if within_bounds:
            score = 1.0
        elif length < self.min_length:
            score = length / self.min_length
        else:
            score = max(0, 1 - (length - self.max_length) / self.max_length)

        return EvalResult(
            passed=within_bounds,
            score=score,
            eval_type=EvalType.CODE_BASED,
            evaluator_name=self.name,
            details={
                "actual_length": length,
                "min_length": self.min_length,
                "max_length": self.max_length if self.max_length != float('inf') else "unlimited",
            },
            reasoning=f"Length {length} is {'within' if within_bounds else 'outside'} bounds [{self.min_length}, {self.max_length}]"
        )


# =============================================================================
# SEMANTIC EVALUATOR
# =============================================================================

class SemanticSimilarityEvaluator(BaseEvaluator):
    """Evaluate similarity using sentence embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.7):
        self.model_name = model_name
        self.threshold = threshold
        self._model = None

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def name(self) -> str:
        return "SemanticSimilarity"

    def evaluate(self, actual: str, expected: str, **kwargs) -> EvalResult:
        """Compare semantic similarity between actual and expected."""
        import numpy as np

        # Generate embeddings
        embeddings = self.model.encode([actual, expected])

        # Calculate cosine similarity
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        # Convert to float and ensure it's between 0 and 1
        similarity = float(max(0, min(1, similarity)))
        passed = similarity >= self.threshold

        return EvalResult(
            passed=passed,
            score=similarity,
            eval_type=EvalType.SEMANTIC,
            evaluator_name=self.name,
            details={
                "similarity": similarity,
                "threshold": self.threshold,
                "model": self.model_name,
            },
            reasoning=f"Semantic similarity: {similarity:.2%} (threshold: {self.threshold:.2%})"
        )

    def batch_evaluate(self, actual: str, candidates: List[str]) -> List[float]:
        """Compare actual against multiple candidates, return similarity scores."""
        import numpy as np

        all_texts = [actual] + candidates
        embeddings = self.model.encode(all_texts)

        actual_emb = embeddings[0]
        similarities = []

        for emb in embeddings[1:]:
            sim = np.dot(actual_emb, emb) / (np.linalg.norm(actual_emb) * np.linalg.norm(emb))
            similarities.append(float(max(0, min(1, sim))))

        return similarities


# =============================================================================
# LLM-AS-JUDGE EVALUATOR
# =============================================================================

class LLMJudgeEvaluator(BaseEvaluator):
    """Use an LLM to evaluate output quality."""

    DEFAULT_CRITERIA = """
    Evaluate the response based on:
    1. Accuracy: Is the information correct?
    2. Relevance: Does it address the question?
    3. Completeness: Is it thorough enough?
    4. Clarity: Is it well-written and clear?
    """

    def __init__(self, llm_client, criteria: str = None):
        self.llm_client = llm_client
        self.criteria = criteria or self.DEFAULT_CRITERIA

    @property
    def name(self) -> str:
        return "LLMJudge"

    def evaluate(
        self,
        actual: str,
        expected: str = None,
        context: str = None,
        question: str = None,
        **kwargs
    ) -> EvalResult:
        """
        Use LLM to judge the quality of actual output.

        Args:
            actual: The LLM output to evaluate
            expected: Optional expected/reference answer
            context: Optional context (e.g., for RAG faithfulness)
            question: The original question/prompt
        """
        prompt = self._build_judge_prompt(actual, expected, context, question)

        system = """You are an expert evaluator. Assess the response and provide:
1. A score from 0.0 to 1.0
2. A brief explanation of your reasoning
3. Specific issues found (if any)

Respond in JSON format:
{
    "score": 0.85,
    "reasoning": "The response is accurate and relevant...",
    "issues": ["Minor issue 1", "Minor issue 2"],
    "strengths": ["Strength 1", "Strength 2"]
}"""

        response = self.llm_client.complete(prompt, system=system, temperature=0.0)

        # Parse the judge's response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                judgment = json.loads(json_match.group(0))
            else:
                judgment = {"score": 0.5, "reasoning": "Could not parse judgment"}
        except json.JSONDecodeError:
            judgment = {"score": 0.5, "reasoning": response.content[:200]}

        score = float(judgment.get("score", 0.5))
        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

        return EvalResult(
            passed=score >= 0.7,  # Default passing threshold
            score=score,
            eval_type=EvalType.LLM_JUDGE,
            evaluator_name=self.name,
            details={
                "issues": judgment.get("issues", []),
                "strengths": judgment.get("strengths", []),
                "judge_model": self.llm_client.get_name(),
                "judge_tokens": response.total_tokens,
                "judge_cost": response.cost_usd,
            },
            reasoning=judgment.get("reasoning", "No reasoning provided")
        )

    def _build_judge_prompt(
        self,
        actual: str,
        expected: str = None,
        context: str = None,
        question: str = None
    ) -> str:
        """Build the evaluation prompt for the judge LLM."""
        parts = [f"## Evaluation Criteria\n{self.criteria}"]

        if question:
            parts.append(f"## Original Question\n{question}")

        if context:
            parts.append(f"## Context/Source Material\n{context[:2000]}")  # Truncate long context

        parts.append(f"## Response to Evaluate\n{actual}")

        if expected:
            parts.append(f"## Reference Answer (for comparison)\n{expected}")

        parts.append("\nPlease evaluate the response and provide your judgment in JSON format.")

        return "\n\n".join(parts)

    def compare(self, response_a: str, response_b: str, question: str = None) -> Dict[str, Any]:
        """Compare two responses and determine which is better."""
        prompt = f"""Compare these two responses and determine which is better.

## Question
{question or 'Not provided'}

## Response A
{response_a}

## Response B
{response_b}

Respond in JSON format:
{{
    "winner": "A" or "B" or "tie",
    "reasoning": "explanation",
    "score_a": 0.0-1.0,
    "score_b": 0.0-1.0
}}"""

        system = "You are an expert evaluator comparing two responses. Be fair and objective."
        response = self.llm_client.complete(prompt, system=system, temperature=0.0)

        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

        return {"winner": "tie", "reasoning": "Could not parse comparison"}


class FaithfulnessEvaluator(BaseEvaluator):
    """Evaluate if output is grounded in provided context (for RAG)."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    @property
    def name(self) -> str:
        return "Faithfulness"

    def evaluate(self, actual: str, expected: str = None, context: str = None, **kwargs) -> EvalResult:
        """
        Check if the response is faithful to the context (no hallucinations).

        Args:
            actual: The LLM response to evaluate
            context: The source context that should ground the response
        """
        if not context:
            return EvalResult(
                passed=False,
                score=0.0,
                eval_type=EvalType.LLM_JUDGE,
                evaluator_name=self.name,
                details={"error": "No context provided"},
                reasoning="Cannot evaluate faithfulness without context"
            )

        prompt = f"""Evaluate if the response is faithful to the given context.

## Context (Source of Truth)
{context[:3000]}

## Response to Evaluate
{actual}

Check each claim in the response:
1. Is it supported by the context?
2. Is it a hallucination (not in context)?
3. Is it a reasonable inference from the context?

Respond in JSON:
{{
    "score": 0.0-1.0,
    "supported_claims": ["claim 1", "claim 2"],
    "hallucinations": ["hallucinated claim 1"],
    "inferences": ["reasonable inference 1"],
    "reasoning": "overall assessment"
}}"""

        system = "You are a faithfulness evaluator. Carefully check if claims are grounded in the context."
        response = self.llm_client.complete(prompt, system=system, temperature=0.0)

        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                judgment = json.loads(json_match.group(0))
            else:
                judgment = {"score": 0.5, "reasoning": "Could not parse"}
        except json.JSONDecodeError:
            judgment = {"score": 0.5, "reasoning": response.content[:200]}

        score = float(judgment.get("score", 0.5))

        return EvalResult(
            passed=score >= 0.8,  # Higher threshold for faithfulness
            score=score,
            eval_type=EvalType.LLM_JUDGE,
            evaluator_name=self.name,
            details={
                "supported_claims": judgment.get("supported_claims", []),
                "hallucinations": judgment.get("hallucinations", []),
                "inferences": judgment.get("inferences", []),
            },
            reasoning=judgment.get("reasoning", "No reasoning provided")
        )


# =============================================================================
# COMPOSITE EVALUATOR
# =============================================================================

class CompositeEvaluator(BaseEvaluator):
    """Combine multiple evaluators with weighted scoring."""

    def __init__(self, evaluators: List[tuple] = None):
        """
        Args:
            evaluators: List of (evaluator, weight) tuples
        """
        self.evaluators = evaluators or []

    @property
    def name(self) -> str:
        return "Composite"

    def add_evaluator(self, evaluator: BaseEvaluator, weight: float = 1.0):
        """Add an evaluator with optional weight."""
        self.evaluators.append((evaluator, weight))
        return self

    def evaluate(self, actual: str, expected: Any = None, **kwargs) -> EvalResult:
        """Run all evaluators and combine scores."""
        if not self.evaluators:
            return EvalResult(
                passed=False,
                score=0.0,
                eval_type=EvalType.CODE_BASED,
                evaluator_name=self.name,
                details={"error": "No evaluators configured"},
                reasoning="Composite evaluator has no sub-evaluators"
            )

        results = []
        total_weight = 0
        weighted_score = 0

        for evaluator, weight in self.evaluators:
            result = evaluator.evaluate(actual, expected, **kwargs)
            results.append({
                "evaluator": evaluator.name,
                "weight": weight,
                "score": result.score,
                "passed": result.passed,
                "reasoning": result.reasoning,
            })
            weighted_score += result.score * weight
            total_weight += weight

        final_score = weighted_score / total_weight if total_weight > 0 else 0
        all_passed = all(r["passed"] for r in results)

        return EvalResult(
            passed=all_passed,
            score=final_score,
            eval_type=EvalType.CODE_BASED,
            evaluator_name=self.name,
            details={"sub_evaluations": results},
            reasoning=f"Composite score: {final_score:.2%} from {len(results)} evaluators"
        )
