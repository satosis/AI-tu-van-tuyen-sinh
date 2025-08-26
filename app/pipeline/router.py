"""
Module định tuyến câu hỏi giữa RAG và LLM.
"""
import logging
from typing import Dict, Any

from app.config import REFLECTION_PROMPT
from app.models.llm import get_llm_model
from app.pipeline.rag_pipeline import get_rag_pipeline

logger = logging.getLogger(__name__)


class SemanticRouter:
    """
    Lớp định tuyến ngữ nghĩa giữa RAG và LLM.
    """

    def __init__(self):
        """
        Khởi tạo semantic router.
        """
        logger.info("Khởi tạo semantic router")

        try:
            # Khởi tạo LLM để phân tích câu hỏi
            self.llm = get_llm_model()

            # Khởi tạo RAG pipeline
            self.rag_pipeline = get_rag_pipeline()

            logger.info("Đã khởi tạo semantic router thành công")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo semantic router: {str(e)}")
            raise

    def route_query(self, query: str, evaluate: bool = False) -> Dict[str, Any]:
        """
        Phân tích và định tuyến câu hỏi.

        Args:
            query (str): Câu hỏi của người dùng
            evaluate (bool, optional): Có đánh giá kết quả không. Mặc định là False.

        Returns:
            Dict[str, Any]: Kết quả xử lý
        """
        try:
            # Phân tích câu hỏi để quyết định sử dụng RAG hay LLM
            logger.info(f"Phân tích câu hỏi: {query}")
            route_decision = self.llm.generate_reflection(query, REFLECTION_PROMPT)

            logger.info(f"Kết quả phân tích: {route_decision}")

            # Chuẩn bị kết quả cơ bản
            result = {
                "query": query,
                "route": route_decision
            }

            # Xử lý theo quyết định định tuyến
            if route_decision == "RAG":
                # Sử dụng RAG để trả lời
                logger.info("Sử dụng RAG để trả lời")
                rag_result = self.rag_pipeline.process(query, evaluate=evaluate)

                # Thu thập thông tin nguồn từ các tài liệu được truy xuất
                sources = ["RAG"]
                for doc in rag_result.get("context", []):
                    src = doc.get("metadata", {}).get("source")
                    if src and src not in sources:
                        sources.append(src)

                # Gộp kết quả
                result.update({
                    "answer": rag_result["answer"],
                    "context": rag_result.get("context", []),
                    "evaluation": rag_result.get("evaluation"),
                    "source": "RAG",
                    "sources": sources
                })

            else:  # LLM
                # Sử dụng LLM trực tiếp để trả lời
                logger.info("Sử dụng LLM để trả lời")
                answer = self.llm.generate_response(query)

                # Gộp kết quả
                result.update({
                    "answer": answer,
                    "context": [],
                    "evaluation": None,
                    "source": "LLM",
                    "sources": []
                })

            logger.info(f"Đã định tuyến và xử lý thành công câu hỏi: {query}")
            return result

        except Exception as e:
            logger.error(f"Lỗi khi định tuyến câu hỏi: {str(e)}")

            # Trả về kết quả lỗi
            error_message = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."

            return {
                "query": query,
                "answer": error_message,
                "context": [],
                "evaluation": None,
                "route": "ERROR",
                "source": "ERROR",
                "sources": [],
                "error": str(e)
            }


# Singleton để giảm chi phí khởi tạo nhiều lần
_semantic_router_instance = None


def get_semantic_router() -> SemanticRouter:
    """
    Trả về instance singleton của semantic router.

    Returns:
        SemanticRouter: Instance semantic router
    """
    global _semantic_router_instance

    if _semantic_router_instance is None:
        _semantic_router_instance = SemanticRouter()

    return _semantic_router_instance
