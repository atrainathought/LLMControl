"""
Structured Outputs Demo Page
"""

import streamlit as st
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(page_title="Structured Outputs | LLMControl", page_icon="📋", layout="wide")

apply_styles()
page_header("Structured Outputs", "Get reliable JSON and schema-validated responses from LLMs", "📋")

section_divider()

# Explanation
with st.expander("What are Structured Outputs?", expanded=False):
    st.markdown("""
    **Structured Outputs** ensure LLMs return data in a specific format.

    | Method | How it works | Reliability |
    |--------|--------------|-------------|
    | **JSON mode** | Request JSON in prompt | ~90% |
    | **Schema validation** | Define expected structure | ~95% |
    | **tool_use** | Claude's native format | 99%+ |
    """)

# Interactive Demo
st.markdown("### Interactive Demo")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Input**")

    output_type = st.radio(
        "Output format",
        ["JSON Object", "List/Array", "Nested Structure"],
        horizontal=True
    )

    sample_prompts = {
        "JSON Object": "Extract info: John Smith, 30 years old, engineer at TechCorp",
        "List/Array": "List 3 benefits of exercise",
        "Nested Structure": "Describe a book: '1984' by George Orwell, published 1949, dystopian fiction"
    }

    user_input = st.text_area("Text to structure", sample_prompts[output_type], height=80)

with col2:
    st.markdown("**Expected Schema**")

    schemas = {
        "JSON Object": {
            "name": "string",
            "age": "number",
            "job_title": "string",
            "company": "string"
        },
        "List/Array": {
            "benefits": ["string", "string", "string"]
        },
        "Nested Structure": {
            "book": {
                "title": "string",
                "author": "string",
                "year": "number",
                "genre": "string"
            }
        }
    }

    st.json(schemas[output_type])

if st.button("Generate Structured Output", type="primary", use_container_width=True):
    section_divider()
    st.markdown("### Results")

    if not st.session_state.get("api_key"):
        st.info("Simulated results (add API key for live demo)")

        results = {
            "JSON Object": {
                "name": "John Smith",
                "age": 30,
                "job_title": "engineer",
                "company": "TechCorp"
            },
            "List/Array": {
                "benefits": [
                    "Improves cardiovascular health",
                    "Boosts mood and mental health",
                    "Increases energy levels"
                ]
            },
            "Nested Structure": {
                "book": {
                    "title": "1984",
                    "author": "George Orwell",
                    "year": 1949,
                    "genre": "dystopian fiction"
                }
            }
        }

        result = results[output_type]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Structured Output**")
            st.json(result)

        with col2:
            st.markdown("**Validation**")
            st.success("Schema validated successfully")
            st.metric("Fields extracted", len(result) if isinstance(result, dict) else len(result.get("benefits", [])))
            st.metric("Parse success", "100%")

    else:
        try:
            import os
            os.environ["ANTHROPIC_API_KEY"] = st.session_state["api_key"]

            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))
            from shared.llm_client import AnthropicClient

            client = AnthropicClient()

            schema_str = json.dumps(schemas[output_type], indent=2)
            prompt = f"""Extract information from this text and return ONLY valid JSON matching this schema:

Schema:
{schema_str}

Text: {user_input}

Return only the JSON object, no explanation."""

            with st.spinner("Generating..."):
                response = client.complete(prompt, temperature=0.0)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Structured Output**")
                try:
                    parsed = json.loads(response.content)
                    st.json(parsed)
                except json.JSONDecodeError:
                    st.code(response.content)
                    st.warning("Response was not valid JSON")

            with col2:
                st.markdown("**Metrics**")
                st.metric("Tokens", response.total_tokens)
                st.metric("Cost", f"${response.cost_usd:.6f}")

        except Exception as e:
            st.error(f"Error: {e}")

section_divider()

# Tool Use Example
st.markdown("### tool_use Format (Most Reliable)")

st.markdown("""
Claude's `tool_use` format provides 99%+ reliability for structured outputs.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tool Definition**")
    st.code("""
tools = [{
    "name": "extract_person",
    "description": "Extract person info",
    "input_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "occupation": {"type": "string"}
        },
        "required": ["name", "age"]
    }
}]
    """, language="python")

with col2:
    st.markdown("**Response**")
    st.code("""
{
    "type": "tool_use",
    "name": "extract_person",
    "input": {
        "name": "John Smith",
        "age": 30,
        "occupation": "engineer"
    }
}
    """, language="json")

section_divider()

# Comparison
st.markdown("### Method Comparison")

import pandas as pd

comparison = pd.DataFrame({
    "Method": ["Prompt-based JSON", "Schema in prompt", "tool_use format"],
    "Reliability": ["~85%", "~95%", "99%+"],
    "Complexity": ["Low", "Medium", "Medium"],
    "Best For": ["Simple extractions", "Complex schemas", "Production use"]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

# Code example
with st.expander("Code Example"):
    st.code("""
from shared.llm_client import AnthropicClient
import json

client = AnthropicClient()

# Method 1: JSON in prompt
response = client.complete(
    "Extract as JSON: {name, age, job}\\nText: John, 30, engineer"
)
data = json.loads(response.content)

# Method 2: tool_use (recommended)
response = client.complete_with_tools(
    "Extract person info from: John Smith, 30, engineer",
    tools=[{
        "name": "extract_person",
        "description": "Extract person information",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "job": {"type": "string"}
            },
            "required": ["name", "age", "job"]
        }
    }]
)
# Response is guaranteed to match schema
    """, language="python")

setup_sidebar()
