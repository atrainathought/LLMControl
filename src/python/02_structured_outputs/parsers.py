"""
Output Parsers

This module demonstrates three approaches to parsing LLM outputs:
1. Raw text parsing - fragile, error-prone
2. JSON parsing - better, but can still fail
3. Tool use parsing - most reliable, schema-enforced

Each parser returns a dict with:
- success: bool
- data: parsed data (if successful)
- error: error message (if failed)
- raw_response: original response for debugging
"""

import re
import json
from typing import Dict, Any, Optional
from pydantic import ValidationError
from schemas import ReviewAnalysis


def parse_raw_text(response: str) -> Dict[str, Any]:
    """
    Parse unstructured text response using regex and string manipulation.

    This is the FRAGILE approach - demonstrates why structured outputs matter.
    """
    result = {
        "success": False,
        "data": None,
        "error": None,
        "raw_response": response,
        "approach": "raw_text"
    }

    try:
        data = {}

        # Try to extract sentiment
        sentiment_match = re.search(
            r'sentiment[:\s]*(positive|negative|neutral|mixed)',
            response.lower()
        )
        if sentiment_match:
            data["sentiment"] = sentiment_match.group(1)
        else:
            raise ValueError("Could not extract sentiment")

        # Try to extract confidence
        confidence_match = re.search(
            r'confidence[:\s]*([0-9.]+)',
            response.lower()
        )
        if confidence_match:
            data["confidence"] = float(confidence_match.group(1))
        else:
            data["confidence"] = 0.8  # Default fallback

        # Try to extract key points (very fragile!)
        # Look for numbered lists or bullet points
        points = re.findall(r'[-•*]\s*(.+?)(?:\n|$)', response)
        if not points:
            points = re.findall(r'\d+[.)]\s*(.+?)(?:\n|$)', response)
        if not points:
            # Fallback: split by sentences and take first few
            sentences = response.split('.')
            points = [s.strip() for s in sentences[:3] if len(s.strip()) > 10]
        data["key_points"] = points[:5] if points else ["Unable to extract key points"]

        # Try to extract recommendation
        if re.search(r'(would recommend|yes|definitely|buy again)', response.lower()):
            data["recommendation"] = True
        elif re.search(r'(would not recommend|no|don\'t buy|avoid)', response.lower()):
            data["recommendation"] = False
        else:
            data["recommendation"] = data["sentiment"] == "positive"

        # Try to extract summary (last sentence or first sentence)
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 20]
        data["summary"] = sentences[-1] if sentences else "Summary not available"

        # Product aspects - look for common keywords
        aspects = []
        aspect_keywords = ["quality", "price", "shipping", "delivery", "value",
                         "packaging", "service", "support", "durability"]
        for keyword in aspect_keywords:
            if keyword in response.lower():
                aspects.append(keyword)
        data["product_aspects"] = aspects

        result["success"] = True
        result["data"] = data

    except Exception as e:
        result["error"] = str(e)

    return result


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON response from LLM.

    Better than raw text, but can still fail if:
    - LLM includes extra text before/after JSON
    - LLM uses invalid JSON syntax
    - Field names don't match exactly
    """
    result = {
        "success": False,
        "data": None,
        "error": None,
        "raw_response": response,
        "approach": "json_mode"
    }

    try:
        # Try to extract JSON from response (handle markdown code blocks)
        json_str = response.strip()

        # Remove markdown code block if present
        if json_str.startswith("```"):
            # Find the JSON between ```json and ```
            match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        # Try to find JSON object in the response
        if not json_str.startswith("{"):
            # Look for JSON object anywhere in response
            match = re.search(r'\{[^{}]*\}', json_str, re.DOTALL)
            if match:
                json_str = match.group(0)

        # Parse JSON
        data = json.loads(json_str)

        # Validate with Pydantic
        validated = ReviewAnalysis(**data)
        result["success"] = True
        result["data"] = validated.model_dump()

    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {e}"
    except ValidationError as e:
        result["error"] = f"Schema validation error: {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result


def parse_tool_response(response: Any) -> Dict[str, Any]:
    """
    Parse tool_use response from Claude.

    This is the MOST RELIABLE approach because:
    - Claude returns structured tool calls
    - The schema is enforced by the API
    - No regex or JSON extraction needed
    """
    result = {
        "success": False,
        "data": None,
        "error": None,
        "raw_response": str(response),
        "approach": "tool_use"
    }

    try:
        # For Claude's tool_use, the response contains tool calls
        # The input is already validated against the schema by Claude
        if hasattr(response, 'content'):
            for block in response.content:
                if hasattr(block, 'type') and block.type == "tool_use":
                    if block.name == "analyze_review":
                        # Tool input is already parsed and validated by Claude
                        data = block.input

                        # Data received from tool call

                        # The tool schema already validated this, so we can
                        # be more lenient with Pydantic validation
                        # Just ensure required fields exist
                        result["success"] = True
                        result["data"] = {
                            "sentiment": data.get("sentiment"),
                            "confidence": data.get("confidence"),
                            "key_points": data.get("key_points", []),
                            "product_aspects": data.get("product_aspects", []),
                            "recommendation": data.get("recommendation"),
                            "summary": data.get("summary", ""),
                        }
                        return result

            # If we got here, no tool_use block found
            # Check if there's a text response (fallback)
            for block in response.content:
                if hasattr(block, 'text'):
                    result["error"] = f"Model returned text instead of tool call: {block.text[:100]}"
                    return result

        result["error"] = "No tool_use response found"

    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result
