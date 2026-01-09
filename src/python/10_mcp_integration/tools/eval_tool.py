"""
Evaluation Tool for MCP.

Exposes the Evals Framework as MCP tools.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "08_evals_framework"))


class EvalTool:
    """
    Evaluation tool that can be exposed via MCP.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the eval tool.

        Args:
            llm_client: LLM client for LLM-as-judge evaluations
        """
        self.llm_client = llm_client
        self._initialized = False

    def initialize(self):
        """Initialize evaluators (lazy loading)."""
        if self._initialized:
            return

        from shared.llm_client import AnthropicClient

        if self.llm_client is None:
            self.llm_client = AnthropicClient()

        self._initialized = True

    def exact_match(self, actual: str, expected: str, case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Check if actual exactly matches expected.

        Args:
            actual: The actual output
            expected: The expected output
            case_sensitive: Whether comparison is case-sensitive

        Returns:
            Evaluation result
        """
        from evaluators import ExactMatchEvaluator

        evaluator = ExactMatchEvaluator(case_sensitive=case_sensitive)
        result = evaluator.evaluate(actual, expected)

        return {
            "evaluator": "exact_match",
            "passed": result.passed,
            "score": result.score,
            "reasoning": result.reasoning
        }

    def contains_keywords(self, text: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Check if text contains all keywords.

        Args:
            text: The text to check
            keywords: List of required keywords

        Returns:
            Evaluation result
        """
        from evaluators import ContainsAllEvaluator

        evaluator = ContainsAllEvaluator()
        result = evaluator.evaluate(text, keywords)

        return {
            "evaluator": "contains_keywords",
            "passed": result.passed,
            "score": result.score,
            "found": result.details.get("found", []),
            "missing": result.details.get("missing", []),
            "reasoning": result.reasoning
        }

    def semantic_similarity(self, text1: str, text2: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Compare semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            threshold: Similarity threshold for passing

        Returns:
            Evaluation result with similarity score
        """
        from evaluators import SemanticSimilarityEvaluator

        evaluator = SemanticSimilarityEvaluator(threshold=threshold)
        result = evaluator.evaluate(text1, text2)

        return {
            "evaluator": "semantic_similarity",
            "passed": result.passed,
            "score": result.score,
            "threshold": threshold,
            "reasoning": result.reasoning
        }

    def llm_judge(
        self,
        output: str,
        question: str = None,
        expected: str = None,
        criteria: str = None
    ) -> Dict[str, Any]:
        """
        Use LLM to judge output quality.

        Args:
            output: The output to evaluate
            question: Original question/prompt
            expected: Optional expected answer
            criteria: Custom evaluation criteria

        Returns:
            LLM judge evaluation result
        """
        self.initialize()

        from evaluators import LLMJudgeEvaluator

        evaluator = LLMJudgeEvaluator(self.llm_client, criteria=criteria)
        result = evaluator.evaluate(output, expected=expected, question=question)

        return {
            "evaluator": "llm_judge",
            "passed": result.passed,
            "score": result.score,
            "reasoning": result.reasoning,
            "strengths": result.details.get("strengths", []),
            "issues": result.details.get("issues", []),
            "judge_tokens": result.details.get("judge_tokens", 0)
        }

    def check_faithfulness(self, output: str, context: str) -> Dict[str, Any]:
        """
        Check if output is faithful to context (no hallucinations).

        Args:
            output: The output to check
            context: The source context

        Returns:
            Faithfulness evaluation result
        """
        self.initialize()

        from evaluators import FaithfulnessEvaluator

        evaluator = FaithfulnessEvaluator(self.llm_client)
        result = evaluator.evaluate(output, context=context)

        return {
            "evaluator": "faithfulness",
            "passed": result.passed,
            "score": result.score,
            "supported_claims": result.details.get("supported_claims", []),
            "hallucinations": result.details.get("hallucinations", []),
            "reasoning": result.reasoning
        }

    def validate_json(self, text: str, schema: Dict[str, type] = None) -> Dict[str, Any]:
        """
        Validate that text is valid JSON (optionally matching schema).

        Args:
            text: Text to validate
            schema: Optional schema dict like {"name": str, "age": int}

        Returns:
            JSON validation result
        """
        from evaluators import JSONSchemaEvaluator

        evaluator = JSONSchemaEvaluator()
        result = evaluator.evaluate(text, schema)

        return {
            "evaluator": "json_schema",
            "passed": result.passed,
            "score": result.score,
            "errors": result.details.get("errors", []),
            "reasoning": result.reasoning
        }

    # MCP tool definitions
    @staticmethod
    def get_tool_definitions() -> list:
        """Return MCP tool definitions for this tool."""
        return [
            {
                "name": "eval_exact_match",
                "description": "Check if two strings match exactly",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "actual": {
                            "type": "string",
                            "description": "The actual output"
                        },
                        "expected": {
                            "type": "string",
                            "description": "The expected output"
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether comparison is case-sensitive",
                            "default": False
                        }
                    },
                    "required": ["actual", "expected"]
                }
            },
            {
                "name": "eval_contains_keywords",
                "description": "Check if text contains all specified keywords",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to check"
                        },
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of required keywords"
                        }
                    },
                    "required": ["text", "keywords"]
                }
            },
            {
                "name": "eval_semantic_similarity",
                "description": "Compare semantic similarity between two texts using embeddings",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text1": {
                            "type": "string",
                            "description": "First text"
                        },
                        "text2": {
                            "type": "string",
                            "description": "Second text"
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Similarity threshold (0-1)",
                            "default": 0.7
                        }
                    },
                    "required": ["text1", "text2"]
                }
            },
            {
                "name": "eval_llm_judge",
                "description": "Use LLM to judge output quality",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "output": {
                            "type": "string",
                            "description": "The output to evaluate"
                        },
                        "question": {
                            "type": "string",
                            "description": "The original question/prompt"
                        },
                        "expected": {
                            "type": "string",
                            "description": "Optional expected answer"
                        },
                        "criteria": {
                            "type": "string",
                            "description": "Custom evaluation criteria"
                        }
                    },
                    "required": ["output"]
                }
            },
            {
                "name": "eval_faithfulness",
                "description": "Check if output is faithful to context (no hallucinations)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "output": {
                            "type": "string",
                            "description": "The output to check"
                        },
                        "context": {
                            "type": "string",
                            "description": "The source context"
                        }
                    },
                    "required": ["output", "context"]
                }
            },
            {
                "name": "eval_json",
                "description": "Validate that text is valid JSON",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to validate as JSON"
                        }
                    },
                    "required": ["text"]
                }
            }
        ]
