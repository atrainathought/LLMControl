"""
RAG Demo Page
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from styles import apply_styles, page_header, section_divider, setup_sidebar

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

st.set_page_config(page_title="RAG Demo | LLMControl", page_icon="📚", layout="wide")

apply_styles()
page_header("RAG", "Retrieval Augmented Generation - Add knowledge to LLMs", "📚")

section_divider()

# Explanation
with st.expander("What is RAG?", expanded=False):
    st.markdown("""
    **RAG** enhances LLM responses by retrieving relevant documents before generating.

    ```
    Question → Embed → Search → Retrieve Docs → Generate with Context → Answer
    ```

    | Without RAG | With RAG |
    |-------------|----------|
    | Relies on training data | Uses current documents |
    | May hallucinate | Grounded in sources |
    | No citations | Can cite sources |
    """)

# Knowledge Base Preview
with st.expander("Knowledge Base"):
    docs = {
        "Leave Policy": "All employees get 20 days annual leave, 10 days sick leave.",
        "IT Security": "Passwords must be 12+ characters, expire every 90 days.",
        "Expenses": "Meal limits: $75/day domestic, $100/day international.",
        "Product Info": "DataSync Pro: $99-499/month, 99.99% uptime SLA.",
        "On-Call": "SEV1 response: 15 minutes. Weekend stipend: $300/day."
    }

    for title, content in docs.items():
        st.markdown(f"**{title}**: {content}")

# Interactive Demo
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Ask a Question")

    questions = [
        "How many days of annual leave do employees get?",
        "What is the password expiration policy?",
        "What is the meal expense limit for domestic travel?",
        "What is the uptime SLA for DataSync Pro?",
        "Custom question..."
    ]

    question_choice = st.selectbox("Select a question", questions)

    if question_choice == "Custom question...":
        question = st.text_input("Enter your question")
    else:
        question = question_choice

    use_rag = st.toggle("Enable RAG", value=True)

    run_demo = st.button("Search & Answer", type="primary", use_container_width=True)

with col2:
    st.markdown("### How it works")

    if use_rag:
        st.success("""
        **RAG Enabled**

        1. Embed your question
        2. Search vector database
        3. Retrieve relevant chunks
        4. Generate answer with context
        """)
    else:
        st.warning("""
        **RAG Disabled**

        1. Send question directly to LLM
        2. LLM uses only training data
        3. May hallucinate or refuse
        """)

# Results
if run_demo and question:
    section_divider()
    st.markdown("### Results")

    if not st.session_state.get("api_key"):
        st.info("Simulated results (add API key for live demo)")

        if use_rag:
            st.markdown("**Retrieved Documents**")

            retrieved = [
                {"doc": "Leave Policy", "similarity": 0.92, "content": "All employees get 20 days annual leave..."},
                {"doc": "Leave Policy", "similarity": 0.78, "content": "Leave accrues at 1.67 days per month..."},
            ]

            for doc in retrieved:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{doc['doc']}**: {doc['content']}")
                with col2:
                    st.metric("Similarity", f"{doc['similarity']:.0%}")

            st.markdown("**Answer**")
            st.success("According to the Leave Policy, employees receive **20 days of annual leave** per year.")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", "100%", "+100%")
            with col2:
                st.metric("Sources", "2 docs")
            with col3:
                st.metric("Confidence", "High")

        else:
            st.markdown("**Answer (No RAG)**")
            st.error("I don't have specific information about your company's leave policy...")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", "0%", "-100%")
            with col2:
                st.metric("Sources", "None")
            with col3:
                st.metric("Confidence", "Low")

    else:
        try:
            import os
            os.environ["ANTHROPIC_API_KEY"] = st.session_state["api_key"]

            sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python" / "04_rag"))

            from shared.llm_client import AnthropicClient
            from documents import KNOWLEDGE_BASE, chunk_all_documents
            from vectorstore import VectorStore

            client = AnthropicClient()

            if use_rag:
                with st.spinner("Setting up RAG..."):
                    chunks = chunk_all_documents(KNOWLEDGE_BASE, strategy="headers")
                    store = VectorStore(collection_name="streamlit_demo")
                    store.clear()
                    store.add_chunks(chunks)

                with st.spinner("Searching..."):
                    results = store.search(question, n_results=3)

                st.markdown("**Retrieved Documents**")
                for r in results:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{r['metadata'].get('topic', 'Doc')}**: {r['content'][:150]}...")
                    with col2:
                        st.metric("Similarity", f"{r['similarity']:.0%}")

                context = "\n\n".join([r["content"] for r in results])
                prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

            else:
                prompt = question

            with st.spinner("Generating..."):
                response = client.complete(prompt, temperature=0.0)

            st.markdown("**Answer**")
            if use_rag:
                st.success(response.content)
            else:
                st.warning(response.content)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tokens", response.total_tokens)
            with col2:
                st.metric("Latency", f"{response.latency_ms:.0f}ms")
            with col3:
                st.metric("Cost", f"${response.cost_usd:.6f}")

        except Exception as e:
            st.error(f"Error: {e}")

section_divider()

# Comparison
st.markdown("### RAG vs No RAG")

import pandas as pd

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Hallucination Risk", "Citations", "Latency"],
    "Without RAG": ["0%", "High", "No", "Fast"],
    "With RAG": ["100%", "Low", "Yes", "Medium"]
})

st.dataframe(comparison, use_container_width=True, hide_index=True)

# Code example
with st.expander("Code Example"):
    st.code("""
from vectorstore import VectorStore
from documents import KNOWLEDGE_BASE, chunk_all_documents
from shared.llm_client import AnthropicClient

# 1. Chunk documents
chunks = chunk_all_documents(KNOWLEDGE_BASE)

# 2. Create vector store
store = VectorStore()
store.add_chunks(chunks)

# 3. Search
results = store.search("annual leave policy", n_results=3)

# 4. Generate with context
context = "\\n".join([r["content"] for r in results])
prompt = f"Context: {context}\\n\\nQuestion: How many days?"

client = AnthropicClient()
response = client.complete(prompt)
    """, language="python")

setup_sidebar()
