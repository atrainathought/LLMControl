"""
Context Management Demo Page
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(page_title="Context Management | LLMControl", page_icon="📄", layout="wide")

apply_styles()
page_header("Context Management", "Handle documents longer than the context window", "📄")

section_divider()

# Explanation
with st.expander("What is Context Management?", expanded=False):
    st.markdown("""
    **Context Management** handles documents that exceed the LLM's token limit.

    | Model | Context Window | ~Pages |
    |-------|---------------|--------|
    | GPT-3.5 | 16K tokens | ~20 pages |
    | GPT-4 | 128K tokens | ~160 pages |
    | Claude 3 | 200K tokens | ~250 pages |

    When documents exceed these limits, we need strategies to process them.
    """)

# Strategies Overview
st.markdown("### Strategies")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Chunking**")
    st.markdown("""
    Split document into pieces

    - Fixed size chunks
    - Semantic chunks
    - Overlap for context
    """)
    st.metric("Best for", "Large docs")

with col2:
    st.markdown("**Summarization**")
    st.markdown("""
    Compress content

    - Hierarchical summaries
    - Map-reduce pattern
    - Rolling summaries
    """)
    st.metric("Best for", "Overview tasks")

with col3:
    st.markdown("**Sliding Window**")
    st.markdown("""
    Process incrementally

    - Moving context window
    - Accumulate insights
    - Stream processing
    """)
    st.metric("Best for", "Sequential analysis")

section_divider()

# Interactive Demo
st.markdown("### Interactive Demo")

# Sample long document
sample_doc = """# Company Annual Report 2024

## Executive Summary
TechFlow Inc. had a strong year with revenue growth of 25% year-over-year. Key achievements include launching three new products, expanding to 5 new markets, and growing our team to 500 employees.

## Financial Highlights
- Total Revenue: $150 million (up from $120 million)
- Net Income: $22 million (up from $15 million)
- Operating Margin: 18% (up from 15%)
- Cash Position: $45 million

## Product Updates
### DataSync Pro
Our flagship product saw 40% user growth. New features include real-time sync, enhanced security, and mobile apps.

### Analytics Suite
Launched in Q2, already adopted by 200+ enterprise customers. Key features: AI-powered insights, custom dashboards, predictive analytics.

### Integration Hub
New product connecting 200+ services. Strong adoption in financial services sector.

## Market Expansion
Entered 5 new markets: Japan, Germany, Brazil, India, Australia. International revenue now 30% of total.

## Team Growth
Hired 150 new employees. Key hires in engineering, sales, and customer success. Employee satisfaction score: 4.5/5.

## Looking Ahead
2025 priorities: AI features, enterprise sales, APAC expansion. Revenue target: $200 million.
"""

st.text_area("Sample Document", sample_doc, height=200, disabled=True)

col1, col2 = st.columns(2)

with col1:
    strategy = st.radio(
        "Strategy",
        ["Chunking", "Summarization", "Sliding Window"],
        horizontal=True
    )

with col2:
    if strategy == "Chunking":
        chunk_size = st.slider("Chunk size (chars)", 200, 1000, 500)
        overlap = st.slider("Overlap (chars)", 0, 200, 50)
    elif strategy == "Summarization":
        summary_type = st.selectbox("Summary type", ["Hierarchical", "Map-Reduce", "Rolling"])
    else:
        window_size = st.slider("Window size (chars)", 500, 2000, 1000)

question = st.text_input("Question about the document", "What was the revenue growth?")

if st.button("Process Document", type="primary", use_container_width=True):
    section_divider()
    st.markdown("### Processing")

    if strategy == "Chunking":
        st.markdown("**Step 1: Split into chunks**")

        # Simple chunking
        chunks = []
        for i in range(0, len(sample_doc), chunk_size - overlap):
            chunk = sample_doc[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        st.write(f"Created **{len(chunks)} chunks** of ~{chunk_size} chars each")

        with st.expander("View chunks"):
            for i, chunk in enumerate(chunks[:3]):
                st.markdown(f"**Chunk {i+1}:**")
                st.text(chunk[:200] + "...")

        st.markdown("**Step 2: Search relevant chunks**")
        st.success("Found 2 relevant chunks matching query")

        st.markdown("**Step 3: Generate answer**")
        st.success("**Answer:** Revenue grew 25% year-over-year, from $120 million to $150 million.")

    elif strategy == "Summarization":
        st.markdown("**Step 1: Summarize sections**")

        sections = ["Executive Summary", "Financial Highlights", "Product Updates", "Market Expansion"]
        for section in sections:
            st.write(f"- Summarizing: {section}")

        st.markdown("**Step 2: Combine summaries**")
        st.info("""
        **Combined Summary:**
        TechFlow had 25% revenue growth to $150M. Key achievements: 3 new products,
        5 new markets, 500 employees. Strong product adoption across DataSync Pro,
        Analytics Suite, and Integration Hub. 2025 target: $200M.
        """)

        st.markdown("**Step 3: Answer from summary**")
        st.success("**Answer:** Revenue growth was 25%, from $120M to $150M.")

    else:  # Sliding Window
        st.markdown("**Step 1: Initialize window**")
        st.write(f"Window size: {window_size} characters")

        st.markdown("**Step 2: Process incrementally**")

        progress = st.progress(0)
        insights = []

        steps = ["Executive Summary", "Financials", "Products", "Markets", "Team"]
        for i, step in enumerate(steps):
            progress.progress((i + 1) / len(steps))
            st.write(f"- Processing: {step}")

        st.markdown("**Step 3: Accumulate insights**")
        st.info("Accumulated insight: Revenue grew 25% YoY ($120M → $150M)")

        st.markdown("**Step 4: Final answer**")
        st.success("**Answer:** Revenue growth was 25% year-over-year.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Doc length", f"{len(sample_doc):,} chars")
    with col2:
        st.metric("Chunks/Steps", len(chunks) if strategy == "Chunking" else 5)
    with col3:
        st.metric("Answer found", "Yes")

section_divider()

# Strategy Comparison
st.markdown("### Strategy Comparison")

import pandas as pd

comparison = pd.DataFrame({
    "Strategy": ["Chunking", "Summarization", "Sliding Window"],
    "Speed": ["Fast", "Medium", "Slow"],
    "Accuracy": ["High", "Medium", "High"],
    "Cost": ["Low", "High", "Medium"],
    "Best For": ["Q&A, Search", "Overview, Summary", "Analysis, Review"]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

# Code example
with st.expander("Code Example"):
    st.code("""
from context_manager import (
    Chunker, Summarizer, SlidingWindow
)

# Strategy 1: Chunking
chunker = Chunker(chunk_size=500, overlap=50)
chunks = chunker.split(document)
relevant = chunker.search(chunks, query, top_k=3)
answer = llm.complete(f"Context: {relevant}\\nQuestion: {query}")

# Strategy 2: Summarization
summarizer = Summarizer(method="hierarchical")
summary = summarizer.summarize(document, max_length=1000)
answer = llm.complete(f"Summary: {summary}\\nQuestion: {query}")

# Strategy 3: Sliding Window
window = SlidingWindow(size=2000, stride=1500)
insights = []
for chunk in window.iterate(document):
    insight = llm.complete(f"Extract key facts: {chunk}")
    insights.append(insight)
answer = llm.complete(f"Insights: {insights}\\nQuestion: {query}")
    """, language="python")

setup_sidebar()
