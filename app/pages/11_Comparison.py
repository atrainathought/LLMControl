"""
Comparison Page - Multi-Test Benchmark
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

st.set_page_config(page_title="Comparison | LLMControl", page_icon="📊", layout="wide")

apply_styles()
page_header("Technique Comparison", "Multi-test benchmark with weighted scoring", "📊")

section_divider()

# Test Cases Definition
st.markdown("### Test Suite")

st.info("""
Each technique is evaluated across **3 dimensions**: Accuracy (by task type), Cost, and Latency.
""")

tab_tests, tab_dims = st.tabs(["Test Categories", "Evaluation Dimensions"])

with tab_tests:
    test_cases = pd.DataFrame({
        "Category": [
            "Factual Accuracy",
            "Reasoning Accuracy",
            "Format Accuracy",
            "Internal Knowledge",
            "Web Knowledge",
            "Ambiguous Queries"
        ],
        "Description": [
            "Direct Q&A with verifiable answers",
            "Multi-step logic and inference",
            "Output in specific JSON/structure",
            "Company docs, policies, internal data",
            "Real-time web info, current events",
            "Vague queries needing clarification"
        ],
        "Complexity": ["Low", "High", "Medium", "Medium", "High", "High"],
        "Example": [
            "What year was Python released?",
            "If sales grew 20% and costs dropped 10%, what's the net margin change?",
            "Return customer data as {name, email, tier}",
            "What's our company's PTO policy?",
            "What's the current stock price of AAPL?",
            "Tell me about leave options"
        ]
    })
    st.dataframe(test_cases, use_container_width=True, hide_index=True)

with tab_dims:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Accuracy Types**")
        st.markdown("""
        - Factual: Correct facts
        - Reasoning: Logic chains
        - Format: Schema compliance
        """)
    with col2:
        st.markdown("**Knowledge Sources**")
        st.markdown("""
        - Internal: Private docs/data
        - Web: Public real-time info
        - Hybrid: Both sources
        """)
    with col3:
        st.markdown("**Query Properties**")
        st.markdown("""
        - Complexity: Simple → Complex
        - Ambiguity: Clear → Vague
        - Scope: Narrow → Broad
        """)

section_divider()

# Raw scores per test
st.markdown("### Raw Scores by Test")

# Accuracy scores (%) for each technique on each test
raw_scores = {
    "Technique": [
        "Zero-shot",
        "Few-shot",
        "Chain-of-thought",
        "Structured Output",
        "Basic RAG",
        "Agentic RAG",
        "Few-shot + RAG"
    ],
    "Factual": [85, 92, 90, 88, 95, 96, 97],
    "Reasoning": [45, 58, 82, 55, 60, 78, 80],
    "Format": [40, 65, 50, 95, 70, 75, 85],
    "Internal Knowledge": [15, 20, 18, 20, 92, 96, 94],
    "Web Knowledge": [10, 15, 12, 15, 45, 95, 50],
    "Ambiguous": [30, 55, 60, 50, 65, 92, 85],
}

scores_df = pd.DataFrame(raw_scores)

categories = ["Factual", "Reasoning", "Format", "Internal Knowledge", "Web Knowledge", "Ambiguous"]

# Display scores table
display_scores = scores_df.copy()
for cat in categories:
    display_scores[cat] = display_scores[cat].apply(lambda x: f"{x}%")
st.dataframe(display_scores, use_container_width=True, hide_index=True)

section_divider()

# Calculate weighted overall score
weights = {
    "Factual": 0.15,
    "Reasoning": 0.20,
    "Format": 0.10,
    "Internal Knowledge": 0.20,
    "Web Knowledge": 0.15,
    "Ambiguous": 0.20
}

scores_df["Overall Score"] = sum(
    scores_df[cat] * weight for cat, weight in weights.items()
).round(1)

# Add cost and latency
scores_df["Cost/Query"] = [0.0001, 0.0002, 0.0003, 0.0002, 0.0005, 0.0015, 0.0007]
scores_df["Latency (ms)"] = [480, 520, 680, 540, 1200, 2800, 1400]
scores_df["Complexity"] = [1, 2, 2, 2, 3, 4, 3]

# Key Metrics Overview
st.markdown("### Key Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Accuracy**")
    st.caption("Weighted score across all test categories")
    best_acc = scores_df.loc[scores_df["Overall Score"].idxmax()]
    st.metric("Top Performer", best_acc["Technique"], f"{best_acc['Overall Score']:.1f}%")

with col2:
    st.markdown("**Cost**")
    st.caption("Price per query in USD")
    cheapest = scores_df.loc[scores_df["Cost/Query"].idxmin()]
    st.metric("Lowest Cost", cheapest["Technique"], f"${cheapest['Cost/Query']:.4f}")

with col3:
    st.markdown("**Latency**")
    st.caption("Response time in milliseconds")
    fastest = scores_df.loc[scores_df["Latency (ms)"].idxmin()]
    st.metric("Fastest", fastest["Technique"], f"{fastest['Latency (ms)']}ms")

section_divider()

# Overall Rankings
st.markdown("### Rankings")

ranking_df = scores_df[["Technique", "Overall Score", "Cost/Query", "Latency (ms)"]].copy()
ranking_df = ranking_df.sort_values("Overall Score", ascending=False)
ranking_df["Rank"] = range(1, len(ranking_df) + 1)
ranking_df = ranking_df[["Rank", "Technique", "Overall Score", "Cost/Query", "Latency (ms)"]]

# Format for display
display_ranking = ranking_df.copy()
display_ranking["Overall Score"] = display_ranking["Overall Score"].apply(lambda x: f"{x:.1f}%")
display_ranking["Cost/Query"] = display_ranking["Cost/Query"].apply(lambda x: f"${x:.4f}")
display_ranking["Latency (ms)"] = display_ranking["Latency (ms)"].apply(lambda x: f"{x}ms")

st.dataframe(display_ranking, use_container_width=True, hide_index=True)

section_divider()

# Visual Analysis
st.markdown("### Visual Analysis")

tab1, tab2, tab3 = st.tabs(["Radar Chart", "Test Breakdown", "Cost-Benefit"])

with tab1:
    st.markdown("**Capability Profile by Technique**")

    # Multi-select for comparing specific techniques
    technique_options = scores_df["Technique"].tolist()
    selected_techniques = st.multiselect(
        "Select techniques to compare (pick 2-3 for best readability)",
        technique_options,
        default=["Zero-shot", "Agentic RAG"],
        key="radar"
    )

    if not selected_techniques:
        st.warning("Select at least one technique to display")
    else:
        # Distinct colors for better contrast
        colors = ["#667eea", "#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899"]

        fig = go.Figure()

        # Map hex colors to rgba for fill
        fill_colors = [
            "rgba(102, 126, 234, 0.15)",  # #667eea
            "rgba(34, 197, 94, 0.15)",    # #22c55e
            "rgba(245, 158, 11, 0.15)",   # #f59e0b
            "rgba(239, 68, 68, 0.15)",    # #ef4444
            "rgba(139, 92, 246, 0.15)",   # #8b5cf6
            "rgba(6, 182, 212, 0.15)",    # #06b6d4
            "rgba(236, 72, 153, 0.15)",   # #ec4899
        ]

        for i, tech in enumerate(selected_techniques):
            row = scores_df[scores_df["Technique"] == tech].iloc[0]
            color = colors[i % len(colors)]
            fill_color = fill_colors[i % len(fill_colors)]

            fig.add_trace(go.Scatterpolar(
                r=[row[cat] for cat in categories] + [row[categories[0]]],  # Close the loop
                theta=categories + [categories[0]],
                name=tech,
                line=dict(color=color, width=3),
                fill='toself',
                fillcolor=fill_color,
                hovertemplate=f"<b>{tech}</b><br>%{{theta}}: %{{r}}%<extra></extra>"
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    ticktext=["20%", "40%", "60%", "80%", "100%"]
                ),
                angularaxis=dict(
                    tickfont=dict(size=12)
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=500,
            margin=dict(t=30, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Quick comparison summary
        if len(selected_techniques) >= 2:
            st.markdown("**Quick Comparison:**")
            comp_df = scores_df[scores_df["Technique"].isin(selected_techniques)][["Technique", "Overall Score", "Cost/Query", "Latency (ms)"]].copy()
            comp_df["Cost/Query"] = comp_df["Cost/Query"].apply(lambda x: f"${x:.4f}")
            comp_df["Overall Score"] = comp_df["Overall Score"].apply(lambda x: f"{x:.1f}%")
            comp_df["Latency (ms)"] = comp_df["Latency (ms)"].apply(lambda x: f"{x}ms")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("**Performance Heatmap**")

    # Create heatmap data
    heatmap_data = scores_df.set_index("Technique")[categories]

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Test Category", y="Technique", color="Score %"),
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=True
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.markdown("**Key Observations:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Internal Knowledge**: RAG techniques dominate (92-96%)
        - **Web Knowledge**: Only Agentic RAG excels (95%)
        - **Reasoning**: Chain-of-thought leads (82%)
        """)
    with col2:
        st.markdown("""
        - **Format**: Structured Output best (95%)
        - **Ambiguous**: Agentic RAG handles vague queries (92%)
        - Zero-shot fails on knowledge & ambiguity
        """)

with tab3:
    st.markdown("**Cost vs Overall Score**")

    fig = px.scatter(
        scores_df,
        x="Cost/Query",
        y="Overall Score",
        size="Complexity",
        color="Technique",
        size_max=40,
        labels={"Cost/Query": "Cost per Query ($)", "Overall Score": "Overall Score %"},
        hover_data=["Latency (ms)"]
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Calculate value score
    scores_df["Value"] = (scores_df["Overall Score"] / (scores_df["Cost/Query"] * 10000)).round(1)
    best_value = scores_df.loc[scores_df["Value"].idxmax()]

    st.success(f"**Best Value:** {best_value['Technique']} — {best_value['Overall Score']:.1f}% accuracy at ${best_value['Cost/Query']:.4f}/query")

section_divider()

# Technique Strengths
st.markdown("### Technique Strengths & Weaknesses")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Best at Each Test:**")
    for test in categories:
        best_idx = scores_df[test].idxmax()
        best_tech = scores_df.loc[best_idx, "Technique"]
        best_score = scores_df.loc[best_idx, test]
        st.write(f"- **{test}**: {best_tech} ({best_score}%)")

with col2:
    st.markdown("**Biggest Gaps:**")
    for _, row in scores_df.iterrows():
        scores = [row[cat] for cat in categories]
        gap = max(scores) - min(scores)
        if gap > 50:
            weak = categories[scores.index(min(scores))]
            st.write(f"- **{row['Technique']}**: Weak on {weak} ({min(scores)}%)")

section_divider()

# Decision Matrix
st.markdown("### Decision Matrix")

st.markdown("""
| Your Priority | Best Choice | Why |
|---------------|-------------|-----|
| **Low cost, simple tasks** | Zero-shot | 85% factual, $0.0001, 480ms |
| **Complex reasoning** | Chain-of-thought | 82% reasoning accuracy |
| **Strict JSON format** | Structured Output | 95% format compliance |
| **Internal docs/data** | Basic RAG | 92% internal knowledge, good value |
| **Web/real-time data** | Agentic RAG | 95% web knowledge |
| **Vague/ambiguous queries** | Agentic RAG | 92% handles unclear requests |
| **Maximum accuracy** | Few-shot + RAG | Best overall balance |
""")

section_divider()

# Custom Weight Calculator
st.markdown("### Custom Scoring")

st.markdown("Adjust weights based on your use case:")

col1, col2, col3 = st.columns(3)

with col1:
    w_factual = st.slider("Factual Accuracy", 0, 100, 15, 5)
    w_reasoning = st.slider("Reasoning Accuracy", 0, 100, 20, 5)

with col2:
    w_format = st.slider("Format Accuracy", 0, 100, 10, 5)
    w_internal = st.slider("Internal Knowledge", 0, 100, 20, 5)

with col3:
    w_web = st.slider("Web Knowledge", 0, 100, 15, 5)
    w_ambiguous = st.slider("Ambiguous Queries", 0, 100, 20, 5)

total_weight = w_factual + w_reasoning + w_format + w_internal + w_web + w_ambiguous

if total_weight > 0:
    # Normalize weights
    custom_weights = {
        "Factual": w_factual / total_weight,
        "Reasoning": w_reasoning / total_weight,
        "Format": w_format / total_weight,
        "Internal Knowledge": w_internal / total_weight,
        "Web Knowledge": w_web / total_weight,
        "Ambiguous": w_ambiguous / total_weight
    }

    # Recalculate scores
    scores_df["Custom Score"] = sum(
        scores_df[cat] * weight for cat, weight in custom_weights.items()
    ).round(1)

    # Show results
    custom_ranking = scores_df[["Technique", "Custom Score"]].sort_values("Custom Score", ascending=False)

    st.markdown("**Your Custom Ranking:**")

    for i, (_, row) in enumerate(custom_ranking.iterrows(), 1):
        if i == 1:
            st.success(f"🥇 **{row['Technique']}**: {row['Custom Score']:.1f}%")
        elif i == 2:
            st.info(f"🥈 **{row['Technique']}**: {row['Custom Score']:.1f}%")
        elif i == 3:
            st.warning(f"🥉 **{row['Technique']}**: {row['Custom Score']:.1f}%")
        else:
            st.write(f"{i}. {row['Technique']}: {row['Custom Score']:.1f}%")

section_divider()

# Summary
st.markdown("### Summary")

# Get top performer
top = scores_df.loc[scores_df["Overall Score"].idxmax()]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Top Overall", top["Technique"], f"{top['Overall Score']:.1f}%")

with col2:
    best_value_row = scores_df.loc[scores_df["Value"].idxmax()]
    st.metric("Best Value", best_value_row["Technique"], f"${best_value_row['Cost/Query']:.4f}")

with col3:
    cheapest = scores_df.loc[scores_df["Cost/Query"].idxmin()]
    st.metric("Lowest Cost", cheapest["Technique"], f"{cheapest['Overall Score']:.1f}%")

setup_sidebar()
