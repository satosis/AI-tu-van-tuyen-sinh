"""
Module chứa các hàm tách và xử lý văn bản.
"""
import re
import logging
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản trước khi chia nhỏ.

    Args:
        text (str): Văn bản cần làm sạch

    Returns:
        str: Văn bản đã được làm sạch
    """
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)

    # Loại bỏ các ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s.,;:?!\-()\'"/]', '', text)

    # Loại bỏ các dòng trắng
    text = re.sub(r'\n\s*\n', '\n', text)

    return text.strip()


def split_by_heading(text: str) -> List[str]:
    """
    Chia văn bản theo tiêu đề.

    Args:
        text (str): Văn bản cần chia

    Returns:
        List[str]: Danh sách các đoạn văn bản theo tiêu đề
    """
    # Mẫu regex để tìm các tiêu đề (chữ in hoa hoặc số theo sau bởi dấu chấm)
    heading_pattern = r'(?:\n|\A)([A-Z\d][A-Z\d\s.,]+:|\d+\.\s+[A-Z])'

    # Chia văn bản theo tiêu đề
    sections = re.split(heading_pattern, text)

    # Ghép lại tiêu đề với nội dung
    result = []
    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            section = sections[i] + sections[i + 1]
        else:
            section = sections[i]
        result.append(section)

    # Xử lý phần đầu nếu không có tiêu đề
    if sections and sections[0].strip():
        result.insert(0, sections[0])

    return result


def create_smart_chunks(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Tạo chunks thông minh, cố gắng giữ nguyên cấu trúc đoạn và tiêu đề.

    Args:
        text (str): Văn bản cần chia nhỏ
        chunk_size (int, optional): Kích thước mỗi chunk. Mặc định là CHUNK_SIZE.
        chunk_overlap (int, optional): Độ chồng lấp giữa các chunks. Mặc định là CHUNK_OVERLAP.

    Returns:
        List[str]: Danh sách các chunks văn bản
    """
    # Làm sạch văn bản
    clean = clean_text(text)

    # Thử chia theo tiêu đề trước
    sections = split_by_heading(clean)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            section_chunks = splitter.split_text(section)
            chunks.extend(section_chunks)

    logger.info(f"Đã tạo {len(chunks)} chunks thông minh")
    return chunks


def extract_metadata_from_chunk(chunk: str) -> Dict[str, Any]:
    """
    Trích xuất metadata từ chunk.

    Args:
        chunk (str): Chunk văn bản

    Returns:
        Dict[str, Any]: Metadata trích xuất được
    """
    metadata = {
        "length": len(chunk),
        "word_count": len(chunk.split()),
    }

    # Trích xuất tiêu đề nếu có
    title_match = re.search(r'^([A-Z][A-Z\d\s.,]+:|\d+\.\s+[A-Z].+?)(?:\n|$)', chunk)
    if title_match:
        metadata["title"] = title_match.group(1).strip()

    return metadata


def enrich_chunks_with_metadata(chunks: List[str], source_file: str) -> List[Dict[str, Any]]:
    """
    Làm giàu chunks với metadata.

    Args:
        chunks (List[str]): Danh sách các chunks
        source_file (str): Tên file nguồn

    Returns:
        List[Dict[str, Any]]: Danh sách chunks với metadata
    """
    enriched_chunks = []

    for i, chunk in enumerate(chunks):
        metadata = extract_metadata_from_chunk(chunk)
        metadata["source"] = source_file
        metadata["chunk_id"] = i

        enriched_chunks.append({
            "text": chunk,
            "metadata": metadata
        })

    return enriched_chunks