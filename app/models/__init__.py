"""
Package initialization for models.
Khởi tạo và quản lý các mô hình LLM, embedding và evaluation.
"""
import logging

logger = logging.getLogger(__name__)
logger.info("Khởi tạo models package")

# Xuất các module để sử dụng trực tiếp từ package
from app.models.llm import get_llm_model
from app.models.embedding import get_embedding_model
from app.models.evaluation import get_evaluation_model

__all__ = [
    'get_llm_model',
    'get_embedding_model',
    'get_evaluation_model'
]