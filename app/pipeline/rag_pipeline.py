"""
Module pipeline RAG (Retrieval-Augmented Generation).
"""
import logging
from typing import Dict, List, Any, Tuple, Optional

from app.config import TOP_K
from app.vectordb.hybrid_search import get_hybrid_search
from app.models.llm import get_llm_model
from app.models.evaluation import get_evaluation_model

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Lớp xử lý pipeline RAG.
    """

    def __init__(self):
        """
        Khởi tạo pipeline RAG.
        """
        logger.info("Khởi tạo RAG pipeline")

        try:
            # Khởi tạo hybrid search
            self.hybrid_search = get_hybrid_search()

            # Khởi tạo LLM
            self.llm = get_llm_model()

            # Khởi tạo mô hình đánh giá
            self.evaluator = get_evaluation_model()

            logger.info("Đã khởi tạo RAG pipeline thành công")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo RAG pipeline: {str(e)}")
            raise

    def process(self,
                query: str,
                top_k: int = TOP_K,
                evaluate: bool = False) -> Dict[str, Any]:
        """
        Xử lý truy vấn bằng pipeline RAG.

        Args:
            query (str): Câu truy vấn của người dùng
            top_k (int, optional): Số lượng tài liệu lấy ra. Mặc định là TOP_K.
            evaluate (bool, optional): Có đánh giá kết quả không. Mặc định là False.

        Returns:
            Dict[str, Any]: Kết quả xử lý
        """
        try:
            # Bước 1: Truy xuất thông tin từ vector database
            retrieved_docs = self.hybrid_search.search(query, n_results=top_k)

            if not retrieved_docs:
                logger.warning(f"Không tìm thấy tài liệu liên quan cho truy vấn: {query}")
                answer = "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu tuyển sinh. Bạn có thể thử cách khác để đặt câu hỏi hoặc liên hệ trực tiếp với nhà trường."

                return {
                    "query": query,
                    "answer": answer,
                    "context": [],
                    "evaluation": None
                }

            # Bước 2: Sinh câu trả lời từ LLM dựa trên thông tin truy xuất
            answer = self.llm.generate_response(query, retrieved_docs)

            # Bước 3: Đánh giá câu trả lời (nếu được yêu cầu)
            evaluation = None
            if evaluate:
                # Tạo context string từ retrieved_docs
                context = "\n\n".join([doc["text"] for doc in retrieved_docs])
                evaluation = self.evaluator.evaluate_response(query, answer, context)

            # Chuẩn bị kết quả trả về
            result = {
                "query": query,
                "answer": answer,
                "context": retrieved_docs,
                "evaluation": evaluation
            }

            logger.info(f"Đã xử lý thành công truy vấn: {query}")
            return result

        except Exception as e:
            logger.error(f"Lỗi khi xử lý truy vấn với RAG: {str(e)}")

            # Trả về kết quả lỗi
            error_message = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."

            return {
                "query": query,
                "answer": error_message,
                "context": [],
                "evaluation": None,
                "error": str(e)
            }


# Singleton để giảm chi phí khởi tạo nhiều lần
_rag_pipeline_instance = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Trả về instance singleton của RAG pipeline.

    Returns:
        RAGPipeline: Instance RAG pipeline
    """
    global _rag_pipeline_instance

    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline()

    return _rag_pipeline_instance