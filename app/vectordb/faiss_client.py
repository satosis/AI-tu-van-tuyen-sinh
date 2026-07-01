"""Client wrapper for a FAISS vector database."""

import logging
import os
import pickle
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

from app.config import FAISS_INDEX_PATH
from app.models.embedding import get_embedding_model

logger = logging.getLogger(__name__)


class FaissDBClient:
    """Simple wrapper around a FAISS index with metadata storage."""

    def __init__(self, index_path: str = FAISS_INDEX_PATH):
        self.index_path = index_path
        self.metadata_path = f"{index_path}.pkl"
        self.embedding_model = get_embedding_model()
        # get_sentence_embedding_dimension() đã đổi tên thành get_embedding_dimension();
        # dùng tên mới nếu có, fallback về tên cũ để tương thích ngược.
        st_model = self.embedding_model.st_model
        if hasattr(st_model, "get_embedding_dimension"):
            self.dimension = st_model.get_embedding_dimension()
        else:
            self.dimension = st_model.get_sentence_embedding_dimension()

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info("Loading existing FAISS index from disk")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "rb") as f:
                data = pickle.load(f)
            self.texts = data.get("texts", [])
            self.metadatas = data.get("metadatas", [])
            self.ids = data.get("ids", [])
        else:
            logger.info("Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts: List[str] = []
            self.metadatas: List[Dict[str, Any]] = []
            self.ids: List[str] = []

    # ------------------------------------------------------------------
    # Persistence helpers
    def _save(self) -> None:
        """Persist index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metadatas": self.metadatas,
                "ids": self.ids,
            }, f)

    # ------------------------------------------------------------------
    # CRUD operations
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the index."""
        ids = [f"doc_{len(self.ids) + i}" for i in range(len(documents))]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        embeddings = self.embedding_model.get_embeddings(texts)
        self.index.add(np.array(embeddings, dtype="float32"))
        self.ids.extend(ids)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self._save()
        logger.info("Added %s documents to FAISS index", len(documents))

    def query_by_text(self, query_text: str, n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search the index for similar documents."""
        if self.index.ntotal == 0:
            return []
        query_vec = self.embedding_model.get_embeddings(query_text)[0]
        distances, indices = self.index.search(np.array([query_vec], dtype="float32"), n_results)
        docs: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            docs.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx],
                "distance": float(dist),
            })
        return docs

    def get_all_documents(self) -> Dict[str, Any]:
        """Return all stored documents and metadata."""
        return {
            "documents": self.texts,
            "metadatas": self.metadatas,
            "ids": self.ids,
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return basic statistics about the index."""
        return {
            "name": os.path.basename(self.index_path),
            "count": len(self.ids),
            "dimension": self.dimension,
            "has_documents": len(self.ids) > 0,
        }

    def delete_index(self) -> bool:
        """Remove persisted index and reset in-memory data."""
        try:
            if os.path.exists(self.index_path):
                os.remove(self.index_path)
            if os.path.exists(self.metadata_path):
                os.remove(self.metadata_path)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts = []
            self.metadatas = []
            self.ids = []
            return True
        except Exception as e:
            logger.error("Failed to delete index: %s", e)
            return False

    # Backwards compatibility with old API
    def delete_collection(self) -> bool:  # pragma: no cover - wrapper
        return self.delete_index()


_faiss_client_instance: Optional[FaissDBClient] = None


def get_faiss_client() -> FaissDBClient:
    """Return a singleton instance of :class:`FaissDBClient`."""
    global _faiss_client_instance
    if _faiss_client_instance is None:
        _faiss_client_instance = FaissDBClient()
    return _faiss_client_instance

