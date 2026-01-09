"""
Relevance Evaluator for Agentic RAG.

Uses LLM to grade retrieved documents for relevance.
This enables the self-correcting loop.
"""

import re
import json
from typing import List, Dict, Any
from dataclasses import dataclass

from retriever import RetrievedDocument, RetrievalResult


@dataclass
class RelevanceGrade:
    """Grade for a single document."""
    doc_id: str
    score: float  # 0.0 to 1.0
    label: str  # "relevant", "partial", "irrelevant"
    reasoning: str


@dataclass
class RelevanceEvaluation:
    """Evaluation results for a retrieval."""
    query: str
    grades: List[RelevanceGrade]
    overall_relevance: float
    has_sufficient_context: bool
    recommendation: str  # "proceed", "rewrite", "expand", "fallback"


class RelevanceEvaluator:
    """
    Evaluates relevance of retrieved documents using LLM.
    """

    def __init__(self, llm_client, threshold: float = 0.6):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client for grading
            threshold: Minimum average relevance to proceed
        """
        self.llm_client = llm_client
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> RelevanceEvaluation:
        """
        Evaluate relevance of retrieved documents.

        Args:
            query: The original query
            retrieval_result: Retrieved documents

        Returns:
            RelevanceEvaluation with grades and recommendation
        """
        grades = []

        for doc in retrieval_result.documents:
            grade = self._grade_document(query, doc)
            grades.append(grade)

            # Update document with grade
            doc.relevance_score = grade.score
            doc.relevance_label = grade.label

        # Calculate overall metrics
        if grades:
            overall_relevance = sum(g.score for g in grades) / len(grades)
            relevant_count = sum(1 for g in grades if g.label == "relevant")
            has_sufficient = relevant_count >= 1 and overall_relevance >= self.threshold
        else:
            overall_relevance = 0.0
            has_sufficient = False

        # Determine recommendation
        recommendation = self._get_recommendation(overall_relevance, grades)

        return RelevanceEvaluation(
            query=query,
            grades=grades,
            overall_relevance=overall_relevance,
            has_sufficient_context=has_sufficient,
            recommendation=recommendation
        )

    def _grade_document(self, query: str, doc: RetrievedDocument) -> RelevanceGrade:
        """Grade a single document's relevance."""
        prompt = f"""Evaluate if this document is relevant to answering the query.

Query: {query}

Document:
{doc.content[:1500]}

Grade the relevance:
- "relevant" (0.8-1.0): Directly answers or contains key information
- "partial" (0.4-0.7): Contains some useful information
- "irrelevant" (0.0-0.3): Not helpful for answering the query

Respond in JSON format:
{{"score": 0.0-1.0, "label": "relevant/partial/irrelevant", "reasoning": "brief explanation"}}"""

        response = self.llm_client.complete(
            prompt,
            system="You are a relevance evaluator. Grade document relevance accurately.",
            temperature=0.0,
            max_tokens=200
        )

        # Parse response
        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group(0))
                return RelevanceGrade(
                    doc_id=doc.id,
                    score=float(result.get("score", 0.5)),
                    label=result.get("label", "partial"),
                    reasoning=result.get("reasoning", "")
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback based on similarity
        if doc.similarity >= 0.7:
            return RelevanceGrade(doc.id, 0.7, "partial", "Fallback: high similarity")
        else:
            return RelevanceGrade(doc.id, 0.3, "irrelevant", "Fallback: low similarity")

    def _get_recommendation(
        self,
        overall_relevance: float,
        grades: List[RelevanceGrade]
    ) -> str:
        """Determine what action to take based on grades."""
        if not grades:
            return "fallback"

        relevant_count = sum(1 for g in grades if g.label == "relevant")
        partial_count = sum(1 for g in grades if g.label == "partial")

        # Strong relevance - proceed
        if relevant_count >= 2 or (relevant_count >= 1 and overall_relevance >= 0.7):
            return "proceed"

        # Some relevance - might be okay
        if relevant_count >= 1 or (partial_count >= 2 and overall_relevance >= 0.5):
            return "proceed"

        # Weak relevance - try query expansion
        if partial_count >= 1 or overall_relevance >= 0.3:
            return "expand"

        # Poor relevance - rewrite query
        if overall_relevance >= 0.2:
            return "rewrite"

        # Very poor - fallback to web or admit uncertainty
        return "fallback"


class QuickRelevanceEvaluator:
    """
    Faster relevance evaluation using embedding similarity only.
    No LLM calls - good for initial filtering.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> RelevanceEvaluation:
        """Quick evaluation based on similarity scores only."""
        grades = []

        for doc in retrieval_result.documents:
            # Grade based on similarity
            if doc.similarity >= 0.7:
                label = "relevant"
                score = doc.similarity
            elif doc.similarity >= 0.4:
                label = "partial"
                score = doc.similarity
            else:
                label = "irrelevant"
                score = doc.similarity

            grades.append(RelevanceGrade(
                doc_id=doc.id,
                score=score,
                label=label,
                reasoning=f"Similarity: {doc.similarity:.2%}"
            ))

            doc.relevance_score = score
            doc.relevance_label = label

        overall = sum(g.score for g in grades) / len(grades) if grades else 0
        relevant_count = sum(1 for g in grades if g.label == "relevant")

        if relevant_count >= 1 and overall >= self.threshold:
            recommendation = "proceed"
        elif overall >= 0.3:
            recommendation = "rewrite"
        else:
            recommendation = "fallback"

        return RelevanceEvaluation(
            query=query,
            grades=grades,
            overall_relevance=overall,
            has_sufficient_context=relevant_count >= 1,
            recommendation=recommendation
        )
