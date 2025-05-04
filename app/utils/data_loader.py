"""
Module để load và xử lý dữ liệu từ các file PDF.
"""
import os
import logging
from typing import List, Dict, Any, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> str:
    """
    Đọc nội dung từ file PDF.

    Args:
        file_path (str): Đường dẫn đến file PDF

    Returns:
        str: Nội dung văn bản từ PDF
    """
    logger.info(f"Đang đọc file PDF: {file_path}")

    try:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() + "\n"

        logger.info(f"Đã đọc thành công {len(reader.pages)} trang từ {file_path}")
        return text

    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        raise


def load_all_pdfs(directory: str = None) -> Dict[str, str]:
    """
    Đọc tất cả các file PDF trong thư mục.

    Args:
        directory (str, optional): Thư mục chứa các file PDF. Mặc định là RAW_DATA_DIR.

    Returns:
        Dict[str, str]: Dictionary với key là tên file và value là nội dung
    """
    if directory is None:
        directory = RAW_DATA_DIR

    logger.info(f"Đang tìm các file PDF trong thư mục: {directory}")

    pdf_files = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                try:
                    pdf_files[file] = load_pdf(file_path)
                    logger.info(f"Đã đọc file: {file}")
                except Exception as e:
                    logger.error(f"Không thể đọc file {file}: {str(e)}")

    logger.info(f"Đã đọc thành công {len(pdf_files)} file PDF")
    return pdf_files


def split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chia nhỏ văn bản thành các chunks.

    Args:
        text (str): Văn bản cần chia nhỏ
        chunk_size (int, optional): Kích thước mỗi chunk. Mặc định là CHUNK_SIZE.
        chunk_overlap (int, optional): Độ chồng lấp giữa các chunks. Mặc định là CHUNK_OVERLAP.

    Returns:
        List[str]: Danh sách các chunks văn bản
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_text(text)
    logger.info(f"Đã chia văn bản thành {len(chunks)} chunks")

    return chunks


def create_document_chunks(pdf_files: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Tạo chunks cho tất cả các file PDF với metadata.

    Args:
        pdf_files (Dict[str, str]): Dictionary với key là tên file và value là nội dung

    Returns:
        List[Dict[str, Any]]: Danh sách các chunk với metadata
    """
    documents = []

    for file_name, content in pdf_files.items():
        chunks = split_text(content)

        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "metadata": {
                    "source": file_name,
                    "chunk_id": i,
                }
            }
            documents.append(doc)

    logger.info(f"Đã tạo tổng cộng {len(documents)} chunks từ {len(pdf_files)} file")
    return documents