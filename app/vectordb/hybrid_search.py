"""
Module hybrid search kết hợp tìm kiếm vector và BM25.
"""
import logging
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from app.vectordb.faiss_client import get_faiss_client
from app.models.embedding import get_embedding_model

logger = logging.getLogger(__name__)


class HybridSearch:
    """
    Lớp tìm kiếm hybrid kết hợp vector search và BM25.
    """

    def __init__(self):
        """
        Khởi tạo hybrid search.
        """
        logger.info("Khởi tạo hybrid search")

        try:
            # Lấy client FAISS
            self.faiss_client = get_faiss_client()

            # Lấy mô hình embedding
            self.embedding_model = get_embedding_model()

            # Khởi tạo BM25 với TF-IDF
            self.vectorizer = None
            self.bm25_docs = []
            self.bm25_initialized = False

            logger.info("Đã khởi tạo hybrid search thành công")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo hybrid search: {str(e)}")
            raise

    def _initialize_bm25(self) -> None:
        """
        Khởi tạo BM25 (TF-IDF) từ dữ liệu trong FAISS.
        """
        try:
            # Lấy toàn bộ dữ liệu từ FAISS
            results = self.faiss_client.get_all_documents()

            if not results or "documents" not in results or not results["documents"]:
                logger.warning("Không có dữ liệu để khởi tạo BM25")
                self.bm25_initialized = False
                return

            # Lưu trữ tài liệu cho bm25
            self.bm25_docs = []
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if "metadatas" in results and i < len(results["metadatas"]) else {}
                self.bm25_docs.append({
                    "text": doc,
                    "metadata": metadata,
                    "id": results["ids"][i] if "ids" in results and i < len(results["ids"]) else f"doc_{i}"
                })

            # Khởi tạo vectorizer TF-IDF
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"(?u)\b\w+\b",
                ngram_range=(1, 2),
                max_df=0.95,
                min_df=2,
                use_idf=True
            )

            # Fit vectorizer với tất cả tài liệu
            self.vectorizer.fit([doc["text"] for doc in self.bm25_docs])

            self.bm25_initialized = True
            logger.info(f"Đã khởi tạo BM25 với {len(self.bm25_docs)} tài liệu")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo BM25: {str(e)}")
            self.bm25_initialized = False

    def _search_bm25(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm dùng BM25 (TF-IDF).

        Args:
            query (str): Câu truy vấn
            n_results (int, optional): Số lượng kết quả. Mặc định là 5.

        Returns:
            List[Dict[str, Any]]: Danh sách tài liệu kết quả
        """
        # Kiểm tra đã khởi tạo BM25 chưa
        if not self.bm25_initialized or not self.vectorizer:
            logger.warning("BM25 chưa được khởi tạo, đang thử khởi tạo lại")
            self._initialize_bm25()

            if not self.bm25_initialized:
                logger.error("Không thể khởi tạo BM25, trả về kết quả rỗng")
                return []

        try:
            # Chuyển đổi query thành vector tf-idf
            query_vector = self.vectorizer.transform([query])

            # Chuyển đổi tất cả tài liệu thành vector tf-idf
            doc_vectors = self.vectorizer.transform([doc["text"] for doc in self.bm25_docs])

            # Tính toán độ tương đồng giữa query và tất cả tài liệu
            similarity_scores = (query_vector * doc_vectors.T).toarray()[0]

            # Sắp xếp kết quả và lấy top n_results
            top_indices = similarity_scores.argsort()[-n_results:][::-1]

            # Tạo danh sách kết quả
            results = []
            for idx in top_indices:
                doc = self.bm25_docs[idx]
                results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": float(similarity_scores[idx]),
                    "source": "bm25"
                })

            logger.info(f"Đã tìm thấy {len(results)} kết quả từ BM25")
            return results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm với BM25: {str(e)}")
            return []

    def _search_vector(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Tìm kiếm dùng vector (cosine similarity).

        Args:
            query (str): Câu truy vấn
            n_results (int, optional): Số lượng kết quả. Mặc định là 5.

        Returns:
            List[Dict[str, Any]]: Danh sách tài liệu kết quả
        """
        try:
            # Tìm kiếm bằng FAISS
            results = self.faiss_client.query_by_text(query, n_results=n_results)

            # Thêm thông tin nguồn
            for doc in results:
                doc["source"] = "vector"

            logger.info(f"Đã tìm thấy {len(results)} kết quả từ vector search")
            return results

        except Exception as e:
            logger.error(f"Lỗi khi tìm kiếm với vector search: {str(e)}")
            return []

    def _normalize_scores(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chuẩn hóa điểm của các tài liệu từ cả hai phương pháp tìm kiếm.

        Args:
            docs (List[Dict[str, Any]]): Danh sách tài liệu kết quả

        Returns:
            List[Dict[str, Any]]: Danh sách tài liệu với điểm đã chuẩn hóa
        """
        if not docs:
            return []

        # Tách tài liệu theo nguồn
        vector_docs = [doc for doc in docs if doc.get("source") == "vector"]
        bm25_docs = [doc for doc in docs if doc.get("source") == "bm25"]

        # Chuẩn hóa điểm cho vector docs (khoảng cách => điểm tương đồng)
        if vector_docs:
            distances = [doc.get("distance", 0) for doc in vector_docs]
            max_dist = max(distances)
            min_dist = min(distances)

            # Đảm bảo không chia cho 0
            dist_range = max_dist - min_dist
            if dist_range == 0:
                dist_range = 1

            for doc in vector_docs:
                # Chuyển khoảng cách thành điểm tương đồng (càng gần càng tốt)
                if "distance" in doc:
                    doc["score"] = 1 - ((doc["distance"] - min_dist) / dist_range)
                else:
                    doc["score"] = 0.5  # Điểm mặc định

        # Chuẩn hóa điểm cho bm25 docs
        if bm25_docs:
            scores = [doc.get("score", 0) for doc in bm25_docs]
            max_score = max(scores)
            min_score = min(scores)

            # Đảm bảo không chia cho 0
            score_range = max_score - min_score
            if score_range == 0:
                score_range = 1

            for doc in bm25_docs:
                if "score" in doc:
                    doc["score"] = (doc["score"] - min_score) / score_range
                else:
                    doc["score"] = 0.5  # Điểm mặc định

        return vector_docs + bm25_docs

    def search(self, query: str, n_results: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """
        Tìm kiếm hybrid kết hợp vector search và BM25.

        Args:
            query (str): Câu truy vấn
            n_results (int, optional): Số lượng kết quả cuối cùng. Mặc định là 5.
            alpha (float, optional): Trọng số cho vector search (1-alpha cho BM25). Mặc định là 0.7.

        Returns:
            List[Dict[str, Any]]: Danh sách tài liệu kết quả
        """
        try:
            # Khởi tạo BM25 nếu chưa
            if not self.bm25_initialized:
                self._initialize_bm25()

            # Tìm kiếm với cả hai phương pháp
            vector_results = self._search_vector(query, n_results=n_results * 2)
            bm25_results = self._search_bm25(query, n_results=n_results * 2)

            # Gộp kết quả
            all_results = vector_results + bm25_results

            # Chuẩn hóa điểm
            normalized_results = self._normalize_scores(all_results)

            # Tính điểm hybrid
            result_dict = {}
            for doc in normalized_results:
                text = doc["text"]
                source = doc["source"]
                score = doc["score"]

                if text not in result_dict:
                    result_dict[text] = {
                        "text": text,
                        "metadata": doc["metadata"],
                        "vector_score": 0,
                        "bm25_score": 0,
                        "hybrid_score": 0
                    }

                if source == "vector":
                    result_dict[text]["vector_score"] = score
                elif source == "bm25":
                    result_dict[text]["bm25_score"] = score

            # Tính điểm hybrid
            for key in result_dict:
                doc = result_dict[key]
                doc["hybrid_score"] = alpha * doc["vector_score"] + (1 - alpha) * doc["bm25_score"]

            # Sắp xếp theo điểm hybrid
            sorted_results = sorted(
                list(result_dict.values()),
                key=lambda x: x["hybrid_score"],
                reverse=True
            )

            # Lấy top n_results
            top_results = sorted_results[:n_results]

            # Chuyển đổi định dạng kết quả
            final_results = []
            for doc in top_results:
                final_results.append({
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "score": doc["hybrid_score"],
                    "vector_score": doc["vector_score"],
                    "bm25_score": doc["bm25_score"]
                })

            logger.info(f"Hybrid search trả về {len(final_results)} kết quả")
            return final_results

        except Exception as e:
            logger.error(f"Lỗi khi thực hiện hybrid search: {str(e)}")

            # Trong trường hợp lỗi, thử dùng chỉ vector search
            try:
                fallback_results = self._search_vector(query, n_results=n_results)
                logger.info(f"Fallback đến vector search, trả về {len(fallback_results)} kết quả")
                return fallback_results
            except Exception as fallback_e:
                logger.error(f"Cả hybrid và fallback đều lỗi: {str(fallback_e)}")
                return []


# Singleton để giảm chi phí khởi tạo nhiều lần
_hybrid_search_instance = None


def get_hybrid_search() -> HybridSearch:
    """
    Trả về instance singleton của hybrid search.

    Returns:
        HybridSearch: Instance hybrid search
    """
    global _hybrid_search_instance

    if _hybrid_search_instance is None:
        _hybrid_search_instance = HybridSearch()

    return _hybrid_search_instance