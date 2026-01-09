"""
Structured Output Schemas

This module defines the data structures we want LLMs to return.
We'll compare three approaches to getting these structures:
1. Raw text parsing (regex/string manipulation)
2. JSON mode (ask for JSON, parse it)
3. Tool use / function calling (schema-enforced)
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# =============================================================================
# TASK: Product Review Analysis
# Extract structured data from product reviews
# =============================================================================

class Sentiment(str, Enum):
    """Sentiment classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class ReviewAnalysis(BaseModel):
    """Structured analysis of a product review."""

    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        description="Overall sentiment of the review (use 'mixed' for conflicting signals)"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    key_points: List[str] = Field(
        description="List of main points mentioned in the review",
        min_length=1,
        max_length=5
    )
    product_aspects: List[str] = Field(
        description="Product aspects mentioned (e.g., 'quality', 'price', 'shipping')",
        default_factory=list
    )
    recommendation: bool = Field(
        description="Whether the reviewer would recommend the product"
    )
    summary: str = Field(
        description="One-sentence summary of the review",
        max_length=200
    )


class SimpleClassification(BaseModel):
    """Minimal structure for sentiment only."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment classification"
    )


# =============================================================================
# TEST DATA
# =============================================================================

TEST_REVIEWS = [
    {
        "review": "Amazing product! Works perfectly and arrived ahead of schedule. The quality exceeded my expectations. Would definitely buy again!",
        "expected_sentiment": "positive",
        "expected_recommendation": True,
    },
    {
        "review": "Complete waste of money. Broke after one day of use. Customer service was unhelpful. Do not buy this.",
        "expected_sentiment": "negative",
        "expected_recommendation": False,
    },
    {
        "review": "It's okay. Does what it says but nothing impressive. Price is fair. Shipping was slow but product works fine.",
        "expected_sentiment": "neutral",
        "expected_recommendation": True,  # Neutral but functional = soft recommend
    },
    {
        "review": "Five stars! My whole family loves this. Great value for money. Fast delivery and excellent packaging.",
        "expected_sentiment": "positive",
        "expected_recommendation": True,
    },
    {
        "review": "Disappointed. The quality doesn't match the price at all. Feels cheap and flimsy. Returning it.",
        "expected_sentiment": "negative",
        "expected_recommendation": False,
    },
    # EDGE CASES - designed to stress test parsers
    {
        "review": "Meh 🤷‍♂️ kinda works I guess??? lol shipping took 4eva but whatevs it's fine ig",
        "expected_sentiment": "mixed",  # Informal mixed feelings
        "expected_recommendation": True,
        "note": "Informal language with emojis and abbreviations"
    },
    {
        "review": "The product itself is excellent quality (5/5) but the seller's customer service was TERRIBLE (0/5). Mixed feelings overall - great item from a bad vendor.",
        "expected_sentiment": "mixed",  # Mixed signals
        "expected_recommendation": True,
        "note": "Mixed review with conflicting signals"
    },
    {
        "review": "I thought I would hate this based on other reviews but WOW was I wrong! Total game changer. Ignore the negative reviews, they don't know what they're talking about.",
        "expected_sentiment": "positive",
        "expected_recommendation": True,
        "note": "Contains negative words but positive sentiment"
    },
]


# =============================================================================
# PROMPTS
# =============================================================================

RAW_TEXT_PROMPT = """Analyze this product review and provide:
1. Sentiment (positive, negative, or neutral)
2. Confidence (0.0 to 1.0)
3. Key points (list the main points)
4. Product aspects mentioned
5. Would recommend? (yes/no)
6. One-sentence summary

Review: {review}

Analysis:"""


JSON_MODE_PROMPT = """Analyze this product review and return a JSON object with these fields:
- sentiment: "positive", "negative", "neutral", or "mixed" (use mixed for conflicting signals)
- confidence: number from 0.0 to 1.0
- key_points: array of strings (main points from the review)
- product_aspects: array of strings (aspects like "quality", "price", "shipping")
- recommendation: boolean (would the reviewer recommend?)
- summary: string (one-sentence summary)

Review: {review}

Return ONLY valid JSON, no other text:"""


TOOL_USE_PROMPT = """Analyze this product review and extract structured information using the analyze_review tool.

You MUST call the analyze_review tool with your analysis. Extract:
- sentiment (positive/negative/neutral/mixed)
- confidence score (0.0 to 1.0)
- key points from the review
- product aspects mentioned
- whether the reviewer recommends the product
- a one-sentence summary

Review to analyze: "{review}"

Call the analyze_review tool now with your analysis."""


def get_tool_schema() -> dict:
    """Get the tool schema for Claude's tool_use feature."""
    return {
        "name": "analyze_review",
        "description": "Extract structured analysis from a product review",
        "input_schema": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral", "mixed"],
                    "description": "Overall sentiment of the review (use 'mixed' for conflicting signals)"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score from 0.0 to 1.0"
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "List of main points from the review"
                },
                "product_aspects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Product aspects mentioned (quality, price, shipping, etc.)"
                },
                "recommendation": {
                    "type": "boolean",
                    "description": "Whether the reviewer would recommend the product"
                },
                "summary": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "One-sentence summary of the review"
                }
            },
            "required": ["sentiment", "confidence", "key_points", "recommendation", "summary"]
        }
    }
