"""
Prompt Engineering Demo Page
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

st.set_page_config(page_title="Prompt Engineering | LLMControl", page_icon="📝", layout="wide")

apply_styles()
page_header("Prompt Engineering", "Zero-shot, few-shot, and chain-of-thought prompting", "📝")

section_divider()

# Explanation
with st.expander("What is Prompt Engineering?", expanded=False):
    st.markdown("""
    **Prompt Engineering** is the art of crafting effective prompts to guide LLM behavior.

    | Technique | Description | Best For |
    |-----------|-------------|----------|
    | **Zero-shot** | No examples | Simple tasks |
    | **Few-shot** | Include examples | Pattern matching |
    | **Chain-of-thought** | Explain reasoning | Complex reasoning |
    """)

# Interactive demo
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Input")

    technique = st.radio(
        "Technique",
        ["Zero-shot", "Few-shot", "Chain-of-thought"],
        horizontal=True
    )

    sample_texts = {
        "Product Review": "This laptop is amazing! Great battery life and the screen is beautiful.",
        "Mixed Review": "The food was okay. Service was slow but the ambiance was nice.",
        "Negative Review": "Terrible experience. The product broke after one day.",
        "Custom": ""
    }

    text_choice = st.selectbox("Sample text", list(sample_texts.keys()))

    if text_choice == "Custom":
        user_text = st.text_area("Enter text", height=80)
    else:
        user_text = st.text_area("Text to classify", sample_texts[text_choice], height=80)

    run_demo = st.button("Classify Sentiment", type="primary", use_container_width=True)

with col2:
    st.markdown("### Prompt Preview")

    if technique == "Zero-shot":
        prompt = f"""Classify the sentiment as positive, negative, or neutral.

Text: {user_text}

Sentiment:"""
        st.caption("Direct instruction, no examples")

    elif technique == "Few-shot":
        prompt = f"""Classify the sentiment of each text.

Text: "I love this product!"
Sentiment: positive

Text: "It's okay, nothing special."
Sentiment: neutral

Text: "Worst purchase ever."
Sentiment: negative

Text: "{user_text}"
Sentiment:"""
        st.caption("3 examples establish the pattern")

    else:
        prompt = f"""Analyze the sentiment step by step.

Text: "{user_text}"

Step 1: Identify key opinion words.
Step 2: Determine if positive, negative, or neutral.
Step 3: Final answer.

Analysis:"""
        st.caption("Explicit reasoning steps")

    st.code(prompt, language="text")

# Results
if run_demo and user_text:
    section_divider()
    st.markdown("### Results")

    if not st.session_state.get("api_key"):
        st.info("Simulated results (add API key for live demo)")

        results = {
            "Zero-shot": {"sentiment": "positive", "confidence": "75%"},
            "Few-shot": {"sentiment": "positive", "confidence": "95%"},
            "Chain-of-thought": {"sentiment": "positive", "confidence": "98%"}
        }

        result = results[technique]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentiment", result["sentiment"].upper())
        with col2:
            st.metric("Confidence", result["confidence"])
        with col3:
            st.metric("Technique", technique)

    else:
        try:
            from shared.llm_client import AnthropicClient
            import os

            os.environ["ANTHROPIC_API_KEY"] = st.session_state["api_key"]
            client = AnthropicClient()

            with st.spinner("Calling Claude..."):
                response = client.complete(prompt, temperature=0.0, max_tokens=200)

            col1, col2, col3 = st.columns(3)
            with col1:
                response_lower = response.content.lower()
                if "positive" in response_lower:
                    sentiment = "POSITIVE"
                elif "negative" in response_lower:
                    sentiment = "NEGATIVE"
                else:
                    sentiment = "NEUTRAL"
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Tokens", response.total_tokens)
            with col3:
                st.metric("Cost", f"${response.cost_usd:.6f}")

            st.markdown("**Response:**")
            st.write(response.content)

        except Exception as e:
            st.error(f"Error: {e}")

section_divider()

# Comparison
st.markdown("### Technique Comparison")

import pandas as pd

df = pd.DataFrame({
    "Technique": ["Zero-shot", "Few-shot", "Chain-of-thought"],
    "Accuracy": [75, 100, 100],
    "Tokens": [50, 150, 200],
    "Best For": ["Simple tasks", "Pattern matching", "Complex reasoning"]
})

col1, col2 = st.columns([1, 1])

with col1:
    st.dataframe(df, use_container_width=True, hide_index=True)

with col2:
    st.bar_chart(df.set_index("Technique")["Accuracy"])

# Code example
with st.expander("Code Example"):
    st.code("""
from shared.llm_client import AnthropicClient

client = AnthropicClient()

# Zero-shot
response = client.complete("Classify: 'Great product!' →")

# Few-shot
response = client.complete('''
"Love it!" → positive
"Hate it!" → negative
"Great product!" →''')

# Chain-of-thought
response = client.complete('''
Analyze step by step:
1. Find opinion words
2. Determine sentiment
Text: "Great product!"
Analysis:''')
    """, language="python")

setup_sidebar()
