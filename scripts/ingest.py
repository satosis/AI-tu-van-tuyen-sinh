"""Script để xử lý và nạp dữ liệu từ các tài liệu vào FAISS."""
import os
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Thêm đường dẫn gốc vào sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import RAW_DATA_DIR, FAISS_INDEX_PATH
from app.utils.data_loader import load_documents
from app.utils.text_splitter import create_smart_chunks, enrich_chunks_with_metadata
from app.vectordb.faiss_client import FaissDBClient

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingest.log')
    ]
)

logger = logging.getLogger(__name__)


def ingest_documents(input_dir: str = None,
                     index_path: str = FAISS_INDEX_PATH,
                     recreate: bool = False) -> None:
    """
    Xử lý và nạp dữ liệu từ các tài liệu vào FAISS.

    Args:
        input_dir (str, optional): Thư mục chứa các tài liệu. Mặc định là RAW_DATA_DIR.
        index_path (str, optional): Đường dẫn lưu FAISS index. Mặc định là FAISS_INDEX_PATH.
        recreate (bool, optional): Có tạo lại index không. Mặc định là False.
    """
    try:
        if input_dir is None:
            input_dir = RAW_DATA_DIR

        logger.info(f"Bắt đầu quá trình nạp dữ liệu từ {input_dir}")

        # Khởi tạo client FAISS
        faiss_client = FaissDBClient(index_path)

        # Nếu recreate và index đã tồn tại, xóa index cũ
        if recreate:
            logger.info("Xóa index cũ")
            faiss_client.delete_index()
            faiss_client = FaissDBClient(index_path)

        # Kiểm tra số lượng tài liệu đã có
        stats = faiss_client.get_collection_stats()
        logger.info(f"Thông tin index hiện tại: {stats}")

        # Đọc tất cả các tài liệu
        documents = load_documents(input_dir)
        if not documents:
            logger.warning(f"Không tìm thấy tài liệu trong thư mục {input_dir}")
            return

        logger.info(f"Đã đọc {len(documents)} tài liệu")

        # Xử lý từng file và tạo chunks
        all_chunks = []

        for file_name, content in documents.items():
            logger.info(f"Đang xử lý file: {file_name}")

            # Tạo chunks thông minh
            chunks = create_smart_chunks(content)

            # Làm giàu chunks với metadata
            enriched_chunks = enrich_chunks_with_metadata(chunks, file_name)

            all_chunks.extend(enriched_chunks)

        logger.info(f"Đã tạo tổng cộng {len(all_chunks)} chunks từ {len(documents)} tài liệu")

        # Thêm chunks vào FAISS
        faiss_client.add_documents(all_chunks)

        # Kiểm tra sau khi thêm
        stats_after = faiss_client.get_collection_stats()
        logger.info(f"Thông tin index sau khi thêm: {stats_after}")

        logger.info("Đã hoàn thành quá trình nạp dữ liệu thành công")

    except Exception as e:
        logger.error(f"Lỗi khi nạp dữ liệu: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nạp dữ liệu tài liệu vào FAISS")

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Thư mục chứa các tài liệu",
    )

    parser.add_argument(
        "--index",
        type=str,
        default=FAISS_INDEX_PATH,
        help="Đường dẫn lưu FAISS index",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Xóa index cũ nếu đã tồn tại",
    )

    args = parser.parse_args()

    ingest_documents(
        input_dir=args.input,
        index_path=args.index,
        recreate=args.recreate,
    )
