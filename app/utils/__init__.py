"""
Package initialization for utils.
Chứa các tiện ích hỗ trợ cho hệ thống như xử lý dữ liệu và phân tách văn bản.
"""
import logging

logger = logging.getLogger(__name__)
logger.info("Khởi tạo utils package")

# Xuất các module để sử dụng trực tiếp từ package
from app.utils.data_loader import load_pdf, load_all_pdfs, create_document_chunks
from app.utils.text_splitter import clean_text, split_by_heading, create_smart_chunks, enrich_chunks_with_metadata

__all__ = [
    'load_pdf',
    'load_all_pdfs',
    'create_document_chunks',
    'clean_text',
    'split_by_heading',
    'create_smart_chunks',
    'enrich_chunks_with_metadata'
]