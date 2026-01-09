"""
LLMControl - Interactive Learning Platform

A Streamlit app showcasing LLM conditioning techniques and multi-agent patterns.
"""

import streamlit as st
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(
    page_title="LLMControl",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_styles()

# Header
page_header("LLMControl", "Interactive Learning Platform for LLM Techniques", "🧠")

section_divider()

# Key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Python Modules", "10", help="Single LLM conditioning techniques")
with col2:
    st.metric("TypeScript Patterns", "4", help="Multi-agent orchestration patterns")
with col3:
    st.metric("Interactive Demos", "14", help="Runnable examples with real LLM calls")
with col4:
    st.metric("Avg. Improvement", "+45%", help="Average accuracy improvement across techniques")

section_divider()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Learning Path", "Metrics"])

with tab1:
    st.markdown("""
    ### What is LLMControl?

    A comprehensive learning project covering **14 techniques** for working with Large Language Models.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Python Modules")
        st.markdown("*Single LLM conditioning techniques*")

        modules = [
            ("01", "Prompt Engineering", "Zero-shot, few-shot, chain-of-thought"),
            ("02", "Structured Outputs", "JSON schemas, tool_use format"),
            ("03", "Function Calling", "External tools and APIs"),
            ("04", "RAG", "Retrieval Augmented Generation"),
            ("05", "Guardrails", "Safety and validation"),
            ("06", "Context Management", "Long document handling"),
            ("07", "LoRA Fine-tuning", "Parameter-efficient training"),
            ("08", "Evals Framework", "Automated quality assessment"),
            ("09", "Agentic RAG", "Self-correcting retrieval"),
            ("10", "MCP Integration", "Model Context Protocol"),
        ]

        for num, name, desc in modules:
            with st.expander(f"{num}. {name}"):
                st.caption(desc)
                st.code(f"python src/python/{num}_*/demo.py", language="bash")

    with col2:
        st.markdown("#### TypeScript Patterns")
        st.markdown("*Multi-agent orchestration*")

        patterns = [
            ("01", "Sequential Chain", "A → B → C pipeline"),
            ("02", "Parallel Fan-out", "Concurrent processing"),
            ("03", "Router/Dispatcher", "Intelligent routing"),
            ("04", "Iterative Refinement", "Quality loops"),
        ]

        for num, name, desc in patterns:
            with st.expander(f"{num}. {name}"):
                st.caption(desc)
                st.code(f"npx tsx src/typescript/{num}_*/demo.ts", language="bash")

        st.markdown("---")

        st.markdown("#### Quick Start")
        st.code("""
# Set API key
export ANTHROPIC_API_KEY=your_key

# Run any demo
python src/python/01_prompt_engineering/demo.py
        """, language="bash")

with tab2:
    st.markdown("### Recommended Learning Path")

    st.markdown("**Beginner**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**1. Prompt Engineering**\n\nZero-shot, few-shot, CoT")
    with col2:
        st.info("**2. Structured Outputs**\n\nJSON and schemas")
    with col3:
        st.info("**3. Function Calling**\n\nTools and APIs")

    st.markdown("**Intermediate**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.warning("**4. RAG**\n\nExternal knowledge")
    with col2:
        st.warning("**5. Guardrails**\n\nSafety checks")
    with col3:
        st.warning("**6. Context Mgmt**\n\nLong documents")

    st.markdown("**Advanced**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("**7. LoRA**\n\nFine-tuning")
    with col2:
        st.success("**8. Evals**\n\nQuality metrics")
    with col3:
        st.success("**9. Agentic RAG**\n\nSelf-correction")
    with col4:
        st.success("**10. MCP**\n\nTool ecosystem")

with tab3:
    st.markdown("### Performance Metrics")

    import pandas as pd

    st.markdown("**Accuracy Improvements**")

    data = pd.DataFrame({
        "Technique": ["Prompt Engineering", "Structured Outputs", "RAG", "Agentic RAG"],
        "Before": [75, 87.5, 0, 75],
        "After": [100, 100, 100, 100],
        "Gain": ["+25%", "+12.5%", "+100%", "+25%"]
    })
    st.dataframe(data, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Accuracy Chart**")
        chart_data = pd.DataFrame({
            "Technique": ["Prompt", "Structured", "RAG", "Agentic"],
            "Before": [75, 87.5, 0, 75],
            "After": [100, 100, 100, 100]
        })
        st.bar_chart(chart_data.set_index("Technique"))

    with col2:
        st.markdown("**Evaluator Comparison**")
        eval_data = pd.DataFrame({
            "Type": ["Code-based", "Semantic", "LLM Judge"],
            "Cost": ["$0", "$0", "$0.0002"],
            "Speed": ["<1ms", "~10ms", "~2s"]
        })
        st.dataframe(eval_data, use_container_width=True, hide_index=True)

# Sidebar
setup_sidebar()
