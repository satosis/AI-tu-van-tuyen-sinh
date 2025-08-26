"""
Module mô hình embedding cho việc chuyển đổi văn bản thành vector.
"""
import logging
from typing import List, Union
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:  # pragma: no cover - only used when optional dependency missing
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as e:  # pragma: no cover - requires explicit install
        raise ImportError(
            "`langchain_huggingface` is required. Install via `pip install langchain-huggingface`."
        ) from e
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Lớp chứa mô hình embedding để chuyển đổi văn bản thành vector.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Khởi tạo mô hình embedding.

        Args:
            model_name (str, optional): Tên mô hình embedding. Mặc định là EMBEDDING_MODEL.
        """
        logger.info(f"Khởi tạo mô hình embedding: {model_name}")

        try:
            self.model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f"Đã khởi tạo mô hình embedding thành công: {model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình embedding: {str(e)}")
            raise

        # Lưu một phiên bản mô hình sentence-transformers gốc để sử dụng các chức năng nâng cao
        self.st_model = SentenceTransformer(model_name)

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Tạo vector embedding cho văn bản.

        Args:
            texts (Union[str, List[str]]): Văn bản hoặc danh sách văn bản cần tạo embedding

        Returns:
            List[List[float]]: Danh sách các vector embedding
        """
        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self.model.embed_documents(texts)
            logger.debug(f"Đã tạo {len(embeddings)} vector embedding")

            return embeddings

        except Exception as e:
            logger.error(f"Lỗi khi tạo embedding: {str(e)}")
            raise

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Tính độ tương đồng cosine giữa hai văn bản.

        Args:
            text1 (str): Văn bản thứ nhất
            text2 (str): Văn bản thứ hai

        Returns:
            float: Độ tương đồng cosine (từ 0 đến 1)
        """
        try:
            return self.st_model.similarity(text1, text2)
        except Exception as e:
            logger.error(f"Lỗi khi tính độ tương đồng: {str(e)}")
            raise


# Singleton để giảm chi phí khởi tạo mô hình nhiều lần
_embedding_model_instance = None


def get_embedding_model() -> EmbeddingModel:
    """Trả về instance singleton của mô hình embedding."""
    global _embedding_model_instance

    if _embedding_model_instance is None:
        _embedding_model_instance = EmbeddingModel()

    return _embedding_model_instance
