"""
Module chứa các hàm tương tác với ChromaDB.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions

from app.config import CHROMA_PERSIST_DIRECTORY, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class ChromaDBClient:
    """
    Lớp tương tác với ChromaDB.
    """

    def __init__(self,
                 persist_directory: str = CHROMA_PERSIST_DIRECTORY,
                 collection_name: str = CHROMA_COLLECTION_NAME):
        """
        Khởi tạo client ChromaDB.

        Args:
            persist_directory (str, optional): Thư mục lưu dữ liệu. Mặc định là CHROMA_PERSIST_DIRECTORY.
            collection_name (str, optional): Tên collection. Mặc định là CHROMA_COLLECTION_NAME.
        """
        logger.info(
            f"Khởi tạo client ChromaDB với persist_directory={persist_directory}, collection_name={collection_name}")

        try:
            # Khởi tạo client ChromaDB
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Khởi tạo hàm embedding
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )

            # Lấy hoặc tạo collection
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Đã kết nối tới collection: {collection_name}")

            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Đã tạo collection mới: {collection_name}")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo client ChromaDB: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Thêm tài liệu vào collection.

        Args:
            documents (List[Dict[str, Any]]): Danh sách các tài liệu
        """
        try:
            # Chuẩn bị dữ liệu
            ids = [f"doc_{i}" for i in range(len(documents))]
            texts = [doc["text"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            # Thêm vào collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            logger.info(f"Đã thêm {len(documents)} tài liệu vào collection")

        except Exception as e:
            logger.error(f"Lỗi khi thêm tài liệu vào collection: {str(e)}")
            raise

    def query_by_text(self,
                      query_text: str,
                      n_results: int = 5,
                      where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Truy vấn collection bằng văn bản.

        Args:
            query_text (str): Văn bản truy vấn
            n_results (int, optional): Số lượng kết quả. Mặc định là 5.
            where (Optional[Dict[str, Any]], optional): Điều kiện lọc. Mặc định là None.

        Returns:
            List[Dict[str, Any]]: Danh sách tài liệu kết quả
        """
        try:
            # Thực hiện truy vấn
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where
            )

            # Chuyển đổi kết quả thành định dạng dễ sử dụng
            docs = []

            if results["documents"] and len(results["documents"][0]) > 0:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] and len(
                        results["metadatas"][0]) > i else {}
                    distance = results["distances"][0][i] if results["distances"] and len(
                        results["distances"][0]) > i else None

                    docs.append({
                        "text": doc,
                        "metadata": metadata,
                        "distance": distance
                    })

            logger.info(f"Đã truy vấn và tìm thấy {len(docs)} tài liệu")
            return docs

        except Exception as e:
            logger.error(f"Lỗi khi truy vấn collection: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Lấy thông tin thống kê của collection.

        Returns:
            Dict[str, Any]: Thông tin thống kê
        """
        try:
            # Đếm số lượng tài liệu
            count = self.collection.count()

            # Lấy tên của collection
            name = self.collection.name

            # Thử lấy mẫu để kiểm tra cấu trúc
            sample = self.collection.peek(limit=1)

            stats = {
                "name": name,
                "count": count,
                "has_documents": len(sample["documents"]) > 0 if "documents" in sample else False,
                "has_embeddings": len(sample["embeddings"]) > 0 if "embeddings" in sample else False,
            }

            logger.info(f"Thông tin collection: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin collection: {str(e)}")
            return {"error": str(e)}

    def delete_collection(self) -> bool:
        """
        Xóa collection.

        Returns:
            bool: True nếu xóa thành công, False nếu có lỗi
        """
        try:
            self.client.delete_collection(self.collection.name)
            logger.info(f"Đã xóa collection: {self.collection.name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi xóa collection: {str(e)}")
            return False


# Singleton để giảm chi phí khởi tạo nhiều lần
_chroma_client_instance = None


def get_chroma_client() -> ChromaDBClient:
    """
    Trả về instance singleton của client ChromaDB.

    Returns:
        ChromaDBClient: Instance client ChromaDB
    """
    global _chroma_client_instance

    if _chroma_client_instance is None:
        _chroma_client_instance = ChromaDBClient()

    return _chroma_client_instance