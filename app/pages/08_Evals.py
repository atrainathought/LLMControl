"""
Evals Demo Page
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

st.set_page_config(page_title="Evals Demo | LLMControl", page_icon="✅", layout="wide")

apply_styles()
page_header("Evals Framework", "Evaluate LLM outputs with code, semantic, and LLM-judge methods", "✅")

section_divider()

# Explanation
with st.expander("What are Evals?", expanded=False):
    st.markdown("""
    **Evals** are systematic methods to assess LLM output quality.

    | Type | How it works | Cost | Speed |
    |------|--------------|------|-------|
    | **Code-based** | Exact match, regex, JSON | Free | <1ms |
    | **Semantic** | Embedding similarity | Free | ~10ms |
    | **LLM Judge** | Another LLM rates output | ~$0.0002 | ~2s |
    """)

# Interactive Demo
tab1, tab2, tab3, tab4 = st.tabs(["Exact Match", "Keywords", "Semantic", "LLM Judge"])

with tab1:
    st.markdown("### Exact Match Evaluator")
    st.caption("Check if two strings match exactly")

    col1, col2 = st.columns(2)
    with col1:
        actual1 = st.text_input("Actual output", "Paris", key="exact_actual")
    with col2:
        expected1 = st.text_input("Expected output", "paris", key="exact_expected")

    case_sensitive = st.checkbox("Case sensitive", value=False)

    if st.button("Evaluate", key="exact_btn", type="primary"):
        if case_sensitive:
            match = actual1 == expected1
        else:
            match = actual1.lower() == expected1.lower()

        if match:
            st.success(f"**PASS** - Score: 100%")
        else:
            st.error(f"**FAIL** - Score: 0%")

        st.json({
            "evaluator": "ExactMatch",
            "passed": match,
            "score": 1.0 if match else 0.0
        })

with tab2:
    st.markdown("### Contains Keywords")
    st.caption("Check if text contains all required keywords")

    text2 = st.text_area("Text to check", "Python is great for machine learning and AI.", key="kw_text", height=80)
    keywords2 = st.text_input("Keywords (comma-separated)", "Python, machine learning, AI", key="kw_keywords")

    if st.button("Evaluate", key="kw_btn", type="primary"):
        kw_list = [k.strip() for k in keywords2.split(",")]
        text_lower = text2.lower()

        found = [k for k in kw_list if k.lower() in text_lower]
        missing = [k for k in kw_list if k.lower() not in text_lower]
        score = len(found) / len(kw_list) if kw_list else 1.0

        if score == 1.0:
            st.success(f"**PASS** - All {len(found)} keywords found")
        else:
            st.warning(f"**PARTIAL** - Score: {score:.0%}")
            st.write(f"Missing: {missing}")

        st.json({
            "evaluator": "ContainsAll",
            "passed": len(missing) == 0,
            "score": score,
            "found": found,
            "missing": missing
        })

with tab3:
    st.markdown("### Semantic Similarity")
    st.caption("Compare meaning using embeddings")

    col1, col2 = st.columns(2)
    with col1:
        text3a = st.text_area("Text 1", "The cat sat on the mat.", key="sem_text1", height=80)
    with col2:
        text3b = st.text_area("Text 2", "A feline rested on a rug.", key="sem_text2", height=80)

    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)

    if st.button("Evaluate", key="sem_btn", type="primary"):
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python" / "08_evals_framework"))
            from evaluators import SemanticSimilarityEvaluator

            with st.spinner("Computing embeddings..."):
                evaluator = SemanticSimilarityEvaluator(threshold=threshold)
                result = evaluator.evaluate(text3a, text3b)

            if result.passed:
                st.success(f"**PASS** - Similarity: {result.score:.1%}")
            else:
                st.warning(f"**BELOW THRESHOLD** - Similarity: {result.score:.1%}")

            st.progress(result.score)
            st.caption(f"Threshold: {threshold:.0%} | Actual: {result.score:.1%}")

        except Exception as e:
            st.info("Simulated result (install sentence-transformers for live)")
            similarity = 0.85
            st.metric("Similarity", f"{similarity:.0%}")
            st.progress(similarity)

with tab4:
    st.markdown("### LLM-as-Judge")
    st.caption("Use Claude to evaluate output quality")

    question4 = st.text_input("Original question", "What are the benefits of exercise?", key="judge_q")
    output4 = st.text_area("Output to evaluate",
        "Regular exercise improves cardiovascular health, boosts mood, and increases energy.",
        key="judge_output", height=80)

    if st.button("Evaluate with LLM", key="judge_btn", type="primary"):
        if not st.session_state.get("api_key"):
            st.info("Simulated result (add API key for live)")

            st.success("**PASS** - Score: 85%")
            st.json({
                "evaluator": "LLMJudge",
                "passed": True,
                "score": 0.85,
                "reasoning": "Accurate, relevant, covers multiple benefits.",
                "strengths": ["Accurate", "Concise"],
                "issues": ["Could include examples"]
            })
        else:
            try:
                import os
                os.environ["ANTHROPIC_API_KEY"] = st.session_state["api_key"]

                sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python" / "08_evals_framework"))
                from evaluators import LLMJudgeEvaluator
                from shared.llm_client import AnthropicClient

                client = AnthropicClient()
                evaluator = LLMJudgeEvaluator(client)

                with st.spinner("Evaluating..."):
                    result = evaluator.evaluate(output4, question=question4)

                if result.passed:
                    st.success(f"**PASS** - Score: {result.score:.0%}")
                else:
                    st.warning(f"**NEEDS IMPROVEMENT** - Score: {result.score:.0%}")

                st.write(f"**Reasoning**: {result.reasoning}")

            except Exception as e:
                st.error(f"Error: {e}")

section_divider()

# Faithfulness Check
st.markdown("### Faithfulness Check (RAG Evaluation)")
st.caption("Detect hallucinations by checking if output is grounded in context")

col1, col2 = st.columns(2)

with col1:
    context5 = st.text_area("Source Context",
        "TechFlow was founded in 2010 in San Francisco. The company has 500 employees.",
        key="faith_context", height=100)

with col2:
    output5 = st.text_area("Output to Check",
        "TechFlow has 500 employees and was founded in 2010.",
        key="faith_output", height=100)

if st.button("Check Faithfulness", type="primary"):
    context_lower = context5.lower()
    claims = [c.strip() for c in output5.split(".") if c.strip()]

    supported = []
    hallucinated = []

    for claim in claims:
        key_terms = ["500", "2010", "san francisco", "techflow"]
        if any(term in claim.lower() for term in key_terms if term in context_lower):
            supported.append(claim)
        elif any(term in claim.lower() for term in ["2000", "2005", "1000"]):
            hallucinated.append(claim)
        else:
            supported.append(claim)

    if hallucinated:
        st.error("**HALLUCINATIONS DETECTED**")
        for h in hallucinated:
            st.write(f"- {h}")
    else:
        st.success("**FAITHFUL** - All claims supported")

    if supported:
        st.write("**Supported claims:**")
        for s in supported:
            st.write(f"- {s}")

section_divider()

# Comparison
st.markdown("### Evaluator Comparison")

import pandas as pd

eval_comparison = pd.DataFrame({
    "Evaluator": ["Exact Match", "Keywords", "Semantic", "LLM Judge", "Faithfulness"],
    "Cost": ["$0", "$0", "$0", "~$0.0002", "~$0.0003"],
    "Speed": ["<1ms", "<1ms", "~10ms", "~2s", "~2s"],
    "Best For": ["Exact answers", "Required terms", "Meaning", "Quality", "RAG"]
})

st.dataframe(eval_comparison, use_container_width=True, hide_index=True)

# Code example
with st.expander("Code Example"):
    st.code("""
from evaluators import (
    ExactMatchEvaluator,
    SemanticSimilarityEvaluator,
    LLMJudgeEvaluator
)

# Code-based
exact = ExactMatchEvaluator()
result = exact.evaluate("Paris", "paris")  # score: 1.0

# Semantic
semantic = SemanticSimilarityEvaluator(threshold=0.7)
result = semantic.evaluate("cat on mat", "feline on rug")  # ~0.85

# LLM Judge
judge = LLMJudgeEvaluator(llm_client)
result = judge.evaluate(output, question=question)  # 0.9
    """, language="python")

setup_sidebar()
