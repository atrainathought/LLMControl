"""
Tool Definitions and Implementations

This module defines tools that an LLM can call:
1. Calculator - Basic math operations
2. Weather - Simulated weather lookup
3. Database - Simulated product database queries

Each tool has:
- A schema (JSON Schema for Claude's tool_use)
- An implementation (actual Python function)
"""

import json
import math
from typing import Dict, Any, List, Callable
from datetime import datetime


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

def calculator(operation: str, a: float, b: float = None) -> Dict[str, Any]:
    """
    Perform mathematical calculations.

    Supported operations:
    - add, subtract, multiply, divide (require a and b)
    - sqrt, square, abs, sin, cos, tan (require only a)
    """
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        elif operation == "divide":
            if b == 0:
                return {"error": "Division by zero"}
            result = a / b
        elif operation == "sqrt":
            if a < 0:
                return {"error": "Cannot take square root of negative number"}
            result = math.sqrt(a)
        elif operation == "square":
            result = a ** 2
        elif operation == "power":
            result = a ** b
        elif operation == "abs":
            result = abs(a)
        elif operation == "sin":
            result = math.sin(math.radians(a))
        elif operation == "cos":
            result = math.cos(math.radians(a))
        elif operation == "tan":
            result = math.tan(math.radians(a))
        else:
            return {"error": f"Unknown operation: {operation}"}

        return {
            "operation": operation,
            "a": a,
            "b": b,
            "result": round(result, 6) if isinstance(result, float) else result
        }
    except Exception as e:
        return {"error": str(e)}


def get_weather(city: str, units: str = "celsius") -> Dict[str, Any]:
    """
    Get weather for a city (simulated data for demo purposes).
    """
    # Simulated weather data
    weather_data = {
        "new york": {"temp_c": 22, "condition": "Partly cloudy", "humidity": 65},
        "london": {"temp_c": 15, "condition": "Rainy", "humidity": 80},
        "tokyo": {"temp_c": 28, "condition": "Sunny", "humidity": 70},
        "paris": {"temp_c": 18, "condition": "Cloudy", "humidity": 75},
        "sydney": {"temp_c": 25, "condition": "Sunny", "humidity": 55},
        "san francisco": {"temp_c": 16, "condition": "Foggy", "humidity": 85},
    }

    city_lower = city.lower()
    if city_lower not in weather_data:
        return {"error": f"Weather data not available for '{city}'"}

    data = weather_data[city_lower]
    temp = data["temp_c"]

    if units == "fahrenheit":
        temp = round(temp * 9/5 + 32, 1)
        temp_unit = "°F"
    else:
        temp_unit = "°C"

    return {
        "city": city.title(),
        "temperature": f"{temp}{temp_unit}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "timestamp": datetime.now().isoformat()
    }


def query_products(
    category: str = None,
    min_price: float = None,
    max_price: float = None,
    in_stock: bool = None,
    search: str = None
) -> Dict[str, Any]:
    """
    Query a simulated product database.
    """
    # Simulated product database
    products = [
        {"id": 1, "name": "Wireless Headphones", "category": "electronics", "price": 79.99, "stock": 45, "rating": 4.5},
        {"id": 2, "name": "USB-C Cable", "category": "electronics", "price": 12.99, "stock": 200, "rating": 4.2},
        {"id": 3, "name": "Laptop Stand", "category": "electronics", "price": 49.99, "stock": 0, "rating": 4.7},
        {"id": 4, "name": "Running Shoes", "category": "sports", "price": 129.99, "stock": 30, "rating": 4.8},
        {"id": 5, "name": "Yoga Mat", "category": "sports", "price": 29.99, "stock": 75, "rating": 4.4},
        {"id": 6, "name": "Water Bottle", "category": "sports", "price": 19.99, "stock": 150, "rating": 4.6},
        {"id": 7, "name": "Coffee Maker", "category": "kitchen", "price": 89.99, "stock": 20, "rating": 4.3},
        {"id": 8, "name": "Blender", "category": "kitchen", "price": 59.99, "stock": 35, "rating": 4.1},
        {"id": 9, "name": "Knife Set", "category": "kitchen", "price": 149.99, "stock": 15, "rating": 4.9},
    ]

    # Apply filters
    results = products

    if category:
        results = [p for p in results if p["category"] == category.lower()]

    if min_price is not None:
        results = [p for p in results if p["price"] >= min_price]

    if max_price is not None:
        results = [p for p in results if p["price"] <= max_price]

    if in_stock is not None:
        if in_stock:
            results = [p for p in results if p["stock"] > 0]
        else:
            results = [p for p in results if p["stock"] == 0]

    if search:
        search_lower = search.lower()
        results = [p for p in results if search_lower in p["name"].lower()]

    return {
        "query": {
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
            "in_stock": in_stock,
            "search": search
        },
        "count": len(results),
        "products": results
    }


# =============================================================================
# TOOL SCHEMAS (for Claude's tool_use)
# =============================================================================

CALCULATOR_SCHEMA = {
    "name": "calculator",
    "description": "Perform mathematical calculations. Use this for any math operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide", "sqrt", "square", "power", "abs", "sin", "cos", "tan"],
                "description": "The mathematical operation to perform"
            },
            "a": {
                "type": "number",
                "description": "First operand (required for all operations)"
            },
            "b": {
                "type": "number",
                "description": "Second operand (required for add, subtract, multiply, divide, power)"
            }
        },
        "required": ["operation", "a"]
    }
}

WEATHER_SCHEMA = {
    "name": "get_weather",
    "description": "Get current weather for a city. Use this when the user asks about weather conditions.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name (e.g., 'New York', 'London', 'Tokyo')"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units (default: celsius)"
            }
        },
        "required": ["city"]
    }
}

PRODUCTS_SCHEMA = {
    "name": "query_products",
    "description": "Search and filter products from the database. Use this when the user asks about products, prices, or inventory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["electronics", "sports", "kitchen"],
                "description": "Product category to filter by"
            },
            "min_price": {
                "type": "number",
                "description": "Minimum price filter"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price filter"
            },
            "in_stock": {
                "type": "boolean",
                "description": "Filter by stock availability"
            },
            "search": {
                "type": "string",
                "description": "Search term for product name"
            }
        },
        "required": []
    }
}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Map tool names to their implementations
TOOL_IMPLEMENTATIONS: Dict[str, Callable] = {
    "calculator": calculator,
    "get_weather": get_weather,
    "query_products": query_products,
}

# All available tool schemas
ALL_TOOLS = [CALCULATOR_SCHEMA, WEATHER_SCHEMA, PRODUCTS_SCHEMA]


def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name with given inputs.

    Returns the tool's result or an error dict.
    """
    if tool_name not in TOOL_IMPLEMENTATIONS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        func = TOOL_IMPLEMENTATIONS[tool_name]
        result = func(**tool_input)
        return result
    except TypeError as e:
        return {"error": f"Invalid arguments for {tool_name}: {e}"}
    except Exception as e:
        return {"error": f"Tool execution failed: {e}"}


# =============================================================================
# TEST QUERIES
# =============================================================================

TEST_QUERIES = [
    {
        "query": "What is 25 multiplied by 17?",
        "expected_tools": ["calculator"],
        "description": "Simple single-tool math query"
    },
    {
        "query": "What's the weather like in Tokyo?",
        "expected_tools": ["get_weather"],
        "description": "Simple single-tool weather query"
    },
    {
        "query": "Show me electronics products under $50 that are in stock",
        "expected_tools": ["query_products"],
        "description": "Database query with multiple filters"
    },
    {
        "query": "What's the square root of 144, and then double that result?",
        "expected_tools": ["calculator", "calculator"],
        "description": "Multi-step calculation requiring two tool calls"
    },
    {
        "query": "I need to buy sports equipment. What's available under $50? Also, what's the weather in San Francisco - should I exercise outdoors?",
        "expected_tools": ["query_products", "get_weather"],
        "description": "Multi-tool query requiring different tools"
    },
]
