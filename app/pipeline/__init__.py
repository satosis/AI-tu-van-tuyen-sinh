"""
Package initialization for pipeline.
Quản lý các thành phần pipeline xử lý câu hỏi bao gồm RAG pipeline, router và reflection.
"""
import logging

logger = logging.getLogger(__name__)
logger.info("Khởi tạo pipeline package")

# Xuất các module để sử dụng trực tiếp từ package
from app.pipeline.rag_pipeline import get_rag_pipeline
from app.pipeline.router import get_semantic_router
from app.pipeline.reflection import get_reflection

__all__ = [
    'get_rag_pipeline',
    'get_semantic_router',
    'get_reflection'
]