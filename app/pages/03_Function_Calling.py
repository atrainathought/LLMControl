"""
Function Calling Demo Page
"""

import streamlit as st
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(page_title="Function Calling | LLMControl", page_icon="🔧", layout="wide")

apply_styles()
page_header("Function Calling", "Connect LLMs to external tools, APIs, and databases", "🔧")

section_divider()

# Explanation
with st.expander("What is Function Calling?", expanded=False):
    st.markdown("""
    **Function Calling** lets LLMs invoke external tools to perform actions.

    ```
    User Query → LLM decides tool → Execute function → Return result → LLM responds
    ```

    | Use Case | Example |
    |----------|---------|
    | **Data lookup** | Query database, search API |
    | **Calculations** | Math, conversions, dates |
    | **Actions** | Send email, create ticket |
    | **External APIs** | Weather, stocks, maps |
    """)

# Available Tools Demo
st.markdown("### Available Tools")

tools = {
    "get_weather": {
        "description": "Get current weather for a city",
        "parameters": {"city": "string"},
        "example_call": {"city": "San Francisco"},
        "example_result": {"temp": 68, "condition": "Sunny", "humidity": 45}
    },
    "calculate": {
        "description": "Perform mathematical calculations",
        "parameters": {"expression": "string"},
        "example_call": {"expression": "15 * 24 + 100"},
        "example_result": {"result": 460}
    },
    "search_database": {
        "description": "Search employee database",
        "parameters": {"query": "string", "limit": "number"},
        "example_call": {"query": "engineering", "limit": 5},
        "example_result": {"count": 3, "employees": ["Alice", "Bob", "Carol"]}
    },
    "send_notification": {
        "description": "Send a notification message",
        "parameters": {"to": "string", "message": "string"},
        "example_call": {"to": "team@company.com", "message": "Meeting in 10 min"},
        "example_result": {"status": "sent", "id": "notif_123"}
    }
}

col1, col2 = st.columns(2)

for i, (name, tool) in enumerate(tools.items()):
    with col1 if i % 2 == 0 else col2:
        with st.expander(f"**{name}**"):
            st.caption(tool["description"])
            st.markdown("**Parameters:**")
            st.json(tool["parameters"])

section_divider()

# Interactive Demo
st.markdown("### Interactive Demo")

sample_queries = {
    "What's the weather in Tokyo?": "get_weather",
    "Calculate 25% of 480": "calculate",
    "Find all engineers in the database": "search_database",
    "Notify the team about the deployment": "send_notification",
    "Custom query...": None
}

query_choice = st.selectbox("Select a query", list(sample_queries.keys()))

if query_choice == "Custom query...":
    user_query = st.text_input("Enter your query")
else:
    user_query = query_choice

if st.button("Process Query", type="primary", use_container_width=True):
    section_divider()
    st.markdown("### Execution Flow")

    # Step 1: LLM Analysis
    st.markdown("**Step 1: LLM Analyzes Query**")
    st.info(f"Query: \"{user_query}\"")

    # Determine which tool to call
    expected_tool = sample_queries.get(query_choice)
    if expected_tool is None:
        # Simple matching for custom queries
        if "weather" in user_query.lower():
            expected_tool = "get_weather"
        elif "calculat" in user_query.lower() or "%" in user_query:
            expected_tool = "calculate"
        elif "find" in user_query.lower() or "search" in user_query.lower():
            expected_tool = "search_database"
        elif "notify" in user_query.lower() or "send" in user_query.lower():
            expected_tool = "send_notification"
        else:
            expected_tool = "calculate"

    tool = tools[expected_tool]

    # Step 2: Tool Selection
    st.markdown("**Step 2: Tool Selection**")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"Selected: **{expected_tool}**")
    with col2:
        st.caption(tool["description"])

    # Step 3: Function Call
    st.markdown("**Step 3: Function Call**")
    st.code(json.dumps({
        "tool": expected_tool,
        "arguments": tool["example_call"]
    }, indent=2), language="json")

    # Step 4: Execution
    st.markdown("**Step 4: Execution Result**")
    st.json(tool["example_result"])

    # Step 5: Final Response
    st.markdown("**Step 5: LLM Response**")

    responses = {
        "get_weather": f"The weather in {tool['example_call'].get('city', 'the city')} is currently {tool['example_result']['condition']} with a temperature of {tool['example_result']['temp']}°F and {tool['example_result']['humidity']}% humidity.",
        "calculate": f"The result of {tool['example_call']['expression']} is **{tool['example_result']['result']}**.",
        "search_database": f"Found {tool['example_result']['count']} employees matching your query: {', '.join(tool['example_result']['employees'])}.",
        "send_notification": f"Notification sent successfully! (ID: {tool['example_result']['id']})"
    }

    st.success(responses[expected_tool])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tool calls", "1")
    with col2:
        st.metric("Execution time", "~1.5s")
    with col3:
        st.metric("Status", "Success")

section_divider()

# Tool Definition
st.markdown("### Defining Tools")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Tool Schema**")
    st.code("""
tools = [{
    "name": "get_weather",
    "description": "Get weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    }
}]
    """, language="python")

with col2:
    st.markdown("**Handler Function**")
    st.code("""
def get_weather(city: str) -> dict:
    # Call weather API
    response = requests.get(
        f"api.weather.com/{city}"
    )
    return response.json()

# Register handler
handlers = {
    "get_weather": get_weather
}
    """, language="python")

section_divider()

# Best Practices
st.markdown("### Best Practices")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Do**")
    st.markdown("""
    - Clear, descriptive tool names
    - Detailed parameter descriptions
    - Handle errors gracefully
    - Validate inputs before execution
    - Log all tool calls
    """)

with col2:
    st.markdown("**Don't**")
    st.markdown("""
    - Expose sensitive operations
    - Skip input validation
    - Return raw error messages
    - Allow unbounded operations
    - Trust LLM outputs blindly
    """)

# Code example
with st.expander("Code Example"):
    st.code("""
from shared.llm_client import AnthropicClient

client = AnthropicClient()

# Define tools
tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
}]

# Define handlers
def get_weather(city: str) -> dict:
    # Real implementation would call weather API
    return {"temp": 72, "condition": "Sunny"}

handlers = {"get_weather": get_weather}

# Process query with tools
response = client.complete_with_tools(
    "What's the weather in Paris?",
    tools=tools
)

# Execute if tool was called
if response.tool_calls:
    for call in response.tool_calls:
        result = handlers[call.name](**call.arguments)
        print(f"Tool result: {result}")
    """, language="python")

setup_sidebar()
