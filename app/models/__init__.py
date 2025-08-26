"""
Package initialization for models.
Khởi tạo và quản lý các mô hình LLM, embedding và evaluation.
"""
import logging

logger = logging.getLogger(__name__)

# Logging should only be emitted once handlers are configured by the
# application.  Emitting a log record at import time can trigger warnings when
# this package is imported before logging is set up by an entry point script,
# so we avoid logging here.

# Xuất các module để sử dụng trực tiếp từ package
from app.models.llm import get_llm_model
from app.models.embedding import get_embedding_model
from app.models.evaluation import get_evaluation_model

__all__ = [
    'get_llm_model',
    'get_embedding_model',
    'get_evaluation_model'
]
