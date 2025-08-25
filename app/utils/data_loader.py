"""
Module để load và xử lý dữ liệu từ các tài liệu đầu vào.
"""
import os
import logging
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import RAW_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> str:
    """Đọc nội dung từ file PDF sử dụng ``PyPDFLoader``."""
    logger.info(f"Đang đọc file PDF: {file_path}")
    try:
        pages = PyPDFLoader(file_path).load()
        text = "\n".join(page.page_content for page in pages)
        logger.info(f"Đã đọc thành công {len(pages)} trang từ {file_path}")
        return text
    except Exception as e:
        logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        raise


def load_document(file_path: str) -> str:
    """
    Đọc nội dung từ tài liệu theo định dạng bất kỳ.

    Hỗ trợ các định dạng phổ biến như PDF, DOCX, PPTX, MD...
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return load_pdf(file_path)

    loader = UnstructuredFileLoader(file_path)
    pages = loader.load()
    return "\n".join(page.page_content for page in pages)


def load_documents(
    directory: str = None, file_types: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Đọc tất cả tài liệu trong thư mục với nhiều định dạng.

    Args:
        directory (str, optional): Thư mục chứa các tài liệu. Mặc định là ``RAW_DATA_DIR``.
        file_types (List[str], optional): Danh sách phần mở rộng cần đọc.
            Mặc định gồm ``['pdf', 'docx', 'pptx', 'md']``.

    Returns:
        Dict[str, str]: Dictionary với key là tên file và value là nội dung
    """
    if directory is None:
        directory = RAW_DATA_DIR
    if file_types is None:
        file_types = ["pdf", "docx", "pptx", "md"]

    logger.info(
        f"Đang tìm các tài liệu trong thư mục: {directory} với định dạng: {file_types}"
    )

    documents: Dict[str, str] = {}
    for ext in file_types:
        loader_cls = PyPDFLoader if ext == "pdf" else UnstructuredFileLoader
        loader = DirectoryLoader(directory, glob=f"**/*.{ext}", loader_cls=loader_cls)
        for doc in loader.load():
            file_name = os.path.basename(doc.metadata.get("source", ""))
            documents[file_name] = documents.get(file_name, "") + doc.page_content + "\n"

    logger.info(f"Đã đọc thành công {len(documents)} file tài liệu")
    return documents


def load_all_pdfs(directory: str = None) -> Dict[str, str]:
    """Hàm tương thích để đọc toàn bộ PDF trong thư mục."""
    return load_documents(directory, ["pdf"])


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