"""
RAG Tool for MCP.

Exposes the Agentic RAG system as an MCP tool.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "04_rag"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "09_agentic_rag"))


class RAGTool:
    """
    RAG tool that can be exposed via MCP.
    """

    def __init__(self, vectorstore=None, llm_client=None, use_agentic: bool = True):
        """
        Initialize the RAG tool.

        Args:
            vectorstore: Pre-initialized vectorstore (or will create one)
            llm_client: LLM client for generation
            use_agentic: Whether to use agentic RAG (with self-correction)
        """
        self.vectorstore = vectorstore
        self.llm_client = llm_client
        self.use_agentic = use_agentic
        self._rag = None
        self._initialized = False

    def initialize(self):
        """Initialize the RAG system (lazy loading)."""
        if self._initialized:
            return

        from shared.llm_client import AnthropicClient
        from documents import KNOWLEDGE_BASE, chunk_all_documents
        from vectorstore import VectorStore
        from agentic_rag import create_agentic_rag, create_basic_rag

        # Create LLM client if not provided
        if self.llm_client is None:
            self.llm_client = AnthropicClient()

        # Create vectorstore if not provided
        if self.vectorstore is None:
            chunks = chunk_all_documents(KNOWLEDGE_BASE, strategy="headers")
            self.vectorstore = VectorStore(
                collection_name="mcp_rag",
                use_sentence_transformers=True
            )
            self.vectorstore.clear()
            self.vectorstore.add_chunks(chunks)

        # Create RAG system
        if self.use_agentic:
            self._rag = create_agentic_rag(
                self.vectorstore,
                self.llm_client,
                max_iterations=3
            )
        else:
            self._rag = create_basic_rag(self.vectorstore, self.llm_client)

        self._initialized = True

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dict with search results
        """
        self.initialize()

        from retriever import Retriever
        retriever = Retriever(self.vectorstore, n_results=n_results)
        result = retriever.retrieve(query)

        return {
            "query": query,
            "results": [
                {
                    "id": doc.id,
                    "content": doc.content[:500],
                    "similarity": doc.similarity,
                    "metadata": doc.metadata
                }
                for doc in result.documents
            ],
            "retrieval_time_ms": result.retrieval_time_ms
        }

    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.

        Args:
            question: The question to answer

        Returns:
            Dict with answer and metadata
        """
        self.initialize()

        result = self._rag.query(question)

        return {
            "question": question,
            "answer": result.answer,
            "confidence": result.confidence,
            "success": result.success,
            "iterations": result.iterations,
            "sources_used": result.relevant_docs_used,
            "total_tokens": result.total_tokens,
            "latency_ms": result.total_latency_ms
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        self.initialize()

        return {
            "documents_count": self.vectorstore.count(),
            "rag_type": "agentic" if self.use_agentic else "basic",
            "model": self.llm_client.get_name() if self.llm_client else "unknown"
        }

    # MCP tool definitions
    @staticmethod
    def get_tool_definitions() -> list:
        """Return MCP tool definitions for this tool."""
        return [
            {
                "name": "rag_search",
                "description": "Search the knowledge base for relevant documents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 3
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "rag_query",
                "description": "Ask a question and get an answer from the knowledge base using RAG",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to answer"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "rag_stats",
                "description": "Get statistics about the knowledge base",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
