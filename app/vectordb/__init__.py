"""
Package initialization for vectordb.
Quản lý các thành phần liên quan đến cơ sở dữ liệu vector và tìm kiếm hybrid.
"""
import logging

logger = logging.getLogger(__name__)
logger.info("Khởi tạo vectordb package")

# Xuất các module để sử dụng trực tiếp từ package
from app.vectordb.faiss_client import get_faiss_client
from app.vectordb.hybrid_search import get_hybrid_search

__all__ = [
    'get_faiss_client',
    'get_hybrid_search'
]