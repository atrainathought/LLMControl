"""
Vector Store for RAG

This module handles:
1. Embedding generation (using sentence-transformers or simple approach)
2. Vector storage with ChromaDB
3. Similarity search for retrieval
"""

import os
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

from documents import Chunk


class SimpleEmbedder:
    """
    Simple embedding approach using bag-of-words + TF-IDF-like scoring.

    This is a fallback when sentence-transformers isn't available.
    For production, use sentence-transformers or OpenAI embeddings.
    """

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.dimension = 1000  # Fixed dimension for simplicity

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _hash_token(self, token: str) -> int:
        """Hash token to fixed dimension."""
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        embeddings = []

        for text in texts:
            tokens = self._tokenize(text)
            vec = [0.0] * self.dimension

            for token in tokens:
                idx = self._hash_token(token)
                vec[idx] += 1.0

            # Normalize
            norm = sum(v*v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]

            embeddings.append(vec)

        return embeddings


class SentenceTransformerEmbedder:
    """
    Embedding using sentence-transformers library.

    This produces high-quality semantic embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class VectorStore:
    """
    Vector store using ChromaDB for storage and retrieval.
    """

    def __init__(
        self,
        collection_name: str = "rag_demo",
        persist_directory: str = None,
        use_sentence_transformers: bool = True
    ):
        import chromadb

        # Initialize ChromaDB
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Initialize embedder
        if use_sentence_transformers:
            try:
                self.embedder = SentenceTransformerEmbedder()
                print("Using SentenceTransformer embeddings (all-MiniLM-L6-v2)")
            except ImportError:
                print("sentence-transformers not available, using simple embedder")
                self.embedder = SimpleEmbedder()
        else:
            self.embedder = SimpleEmbedder()
            print("Using simple bag-of-words embeddings")

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.collection_name = collection_name

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """
        Add chunks to the vector store.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        metadatas = [
            {**chunk.metadata, "doc_id": chunk.doc_id}
            for chunk in chunks
        ]

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 3,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Returns list of results with content, metadata, and similarity score.
        """
        # Generate query embedding
        query_embedding = self.embedder.embed([query])[0]

        # Search
        where = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted.append({
                    "id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    # Convert distance to similarity (cosine distance to similarity)
                    "similarity": 1 - results['distances'][0][i]
                })

        return formatted

    def clear(self):
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        """Return the number of items in the store."""
        return self.collection.count()


def create_vectorstore_from_documents(
    chunks: List[Chunk],
    collection_name: str = "rag_demo",
    use_sentence_transformers: bool = True
) -> VectorStore:
    """
    Convenience function to create and populate a vector store.
    """
    store = VectorStore(
        collection_name=collection_name,
        use_sentence_transformers=use_sentence_transformers
    )
    store.clear()  # Start fresh
    count = store.add_chunks(chunks)
    print(f"Added {count} chunks to vector store")
    return store
