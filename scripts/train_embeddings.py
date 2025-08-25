"""
Script để huấn luyện hoặc tinh chỉnh mô hình embedding dựa trên dữ liệu tuyển sinh.
"""
import os
import logging
import argparse
from pathlib import Path
import json
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Thêm đường dẫn gốc vào sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import EMBEDDING_MODEL
from app.utils.data_loader import load_documents, create_document_chunks
from app.utils.text_splitter import create_smart_chunks

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_embeddings.log')
    ]
)

logger = logging.getLogger(__name__)


def prepare_training_data(input_dir: str,
                          query_pairs_file: str = None,
                          output_file: str = None) -> List[Dict[str, Any]]:
    """
    Chuẩn bị dữ liệu huấn luyện từ các tài liệu và cặp query.

    Args:
        input_dir (str): Thư mục chứa các tài liệu
        query_pairs_file (str, optional): File chứa các cặp câu hỏi-đáp. Mặc định là None.
        output_file (str, optional): File để lưu dữ liệu đã chuẩn bị. Mặc định là None.

    Returns:
        List[Dict[str, Any]]: Danh sách dữ liệu huấn luyện
    """
    try:
        logger.info(f"Đang chuẩn bị dữ liệu huấn luyện từ thư mục: {input_dir}")

        # Tải và xử lý các tài liệu
        pdf_files = load_documents(input_dir)

        if not pdf_files:
            logger.warning(f"Không tìm thấy tài liệu nào trong thư mục {input_dir}")
            return []

        logger.info(f"Đã tải {len(pdf_files)} tài liệu")

        # Tạo chunks từ các file PDF
        all_chunks = []

        for file_name, content in pdf_files.items():
            logger.info(f"Đang xử lý tài liệu: {file_name}")
            chunks = create_smart_chunks(content)

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk,
                    "source": file_name
                })

        logger.info(f"Đã tạo {len(all_chunks)} chunks từ các tài liệu")

        # Tải các cặp câu hỏi-đáp nếu có
        query_pairs = []

        if query_pairs_file and os.path.exists(query_pairs_file):
            try:
                with open(query_pairs_file, 'r', encoding='utf-8') as f:
                    query_pairs = json.load(f)

                logger.info(f"Đã tải {len(query_pairs)} cặp câu hỏi-đáp từ {query_pairs_file}")
            except Exception as e:
                logger.error(f"Lỗi khi tải cặp câu hỏi-đáp: {str(e)}")

        # Tạo dữ liệu huấn luyện
        training_data = []

        # 1. Thêm dữ liệu từ chunks
        for i, chunk in enumerate(all_chunks):
            training_data.append({
                "text": chunk["text"],
                "type": "document",
                "source": chunk["source"],
                "id": f"doc_{i}"
            })

        # 2. Thêm dữ liệu từ cặp câu hỏi-đáp
        for i, pair in enumerate(query_pairs):
            if "question" in pair and "answer" in pair:
                # Thêm câu hỏi
                training_data.append({
                    "text": pair["question"],
                    "type": "query",
                    "id": f"query_{i}"
                })

                # Tìm kiếm chunks liên quan đến câu trả lời
                if "related_chunks" in pair:
                    for chunk_id in pair["related_chunks"]:
                        training_data.append({
                            "text": all_chunks[chunk_id]["text"] if chunk_id < len(all_chunks) else "",
                            "type": "document",
                            "source": all_chunks[chunk_id]["source"] if chunk_id < len(all_chunks) else "unknown",
                            "id": f"doc_{chunk_id}",
                            "related_query": f"query_{i}"
                        })

        logger.info(f"Đã tạo tổng cộng {len(training_data)} mẫu dữ liệu huấn luyện")

        # Lưu dữ liệu đã chuẩn bị nếu được yêu cầu
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Đã lưu dữ liệu huấn luyện vào {output_file}")

        return training_data

    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị dữ liệu huấn luyện: {str(e)}")
        return []


def create_training_examples(training_data: List[Dict[str, Any]]) -> List[InputExample]:
    """
    Tạo các ví dụ huấn luyện cho Sentence Transformers.

    Args:
        training_data (List[Dict[str, Any]]): Danh sách dữ liệu huấn luyện

    Returns:
        List[InputExample]: Danh sách các ví dụ huấn luyện
    """
    examples = []

    # Tìm các cặp query-document
    queries = [item for item in training_data if item["type"] == "query"]

    for query in queries:
        query_id = query["id"]
        query_text = query["text"]

        # Tìm các tài liệu liên quan đến query này
        related_docs = [item for item in training_data if item.get("related_query") == query_id]

        # Nếu có tài liệu liên quan, tạo các cặp positive
        for doc in related_docs:
            examples.append(InputExample(
                texts=[query_text, doc["text"]],
                label=1.0  # Positive pair
            ))

            # Tạo một số cặp negative
            negative_count = min(3, len(training_data) - len(related_docs))

            if negative_count > 0:
                # Lấy ngẫu nhiên các tài liệu không liên quan
                negative_docs = [
                    item for item in training_data
                    if item["type"] == "document" and not item.get("related_query") == query_id
                ]

                import random
                negative_samples = random.sample(negative_docs, min(negative_count, len(negative_docs)))

                for neg_doc in negative_samples:
                    examples.append(InputExample(
                        texts=[query_text, neg_doc["text"]],
                        label=0.0  # Negative pair
                    ))

    logger.info(f"Đã tạo {len(examples)} ví dụ huấn luyện")
    return examples


def train_embedding_model(data_dir: str,
                          query_pairs_file: str = None,
                          output_dir: str = "./trained_embeddings",
                          model_name: str = EMBEDDING_MODEL,
                          epochs: int = 4,
                          batch_size: int = 16,
                          learning_rate: float = 2e-5,
                          save_intermediate: bool = False) -> None:
    """
    Huấn luyện hoặc tinh chỉnh mô hình embedding.

    Args:
        data_dir (str): Thư mục chứa dữ liệu huấn luyện
        query_pairs_file (str, optional): File chứa các cặp câu hỏi-đáp. Mặc định là None.
        output_dir (str, optional): Thư mục lưu mô hình. Mặc định là "./trained_embeddings".
        model_name (str, optional): Tên mô hình cơ sở. Mặc định là EMBEDDING_MODEL.
        epochs (int, optional): Số epochs huấn luyện. Mặc định là 4.
        batch_size (int, optional): Kích thước batch. Mặc định là 16.
        learning_rate (float, optional): Tốc độ học. Mặc định là 2e-5.
        save_intermediate (bool, optional): Có lưu mô hình trung gian không. Mặc định là False.
    """
    try:
        # Chuẩn bị dữ liệu huấn luyện
        training_data = prepare_training_data(
            data_dir,
            query_pairs_file=query_pairs_file,
            output_file=os.path.join(output_dir, "training_data.json") if output_dir else None
        )

        if not training_data:
            logger.error("Không có dữ liệu huấn luyện")
            return

        # Tạo các ví dụ huấn luyện
        examples = create_training_examples(training_data)

        if not examples:
            logger.warning("Không có ví dụ huấn luyện")
            return

        # Tạo DataLoader
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)

        # Khởi tạo mô hình
        logger.info(f"Khởi tạo mô hình: {model_name}")
        model = SentenceTransformer(model_name)

        # Định nghĩa hàm mất mát (CosineSimilarityLoss)
        train_loss = losses.CosineSimilarityLoss(model)

        # Đảm bảo thư mục output tồn tại
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Huấn luyện mô hình
        logger.info(f"Bắt đầu huấn luyện với {len(examples)} ví dụ, {epochs} epochs")

        # Callback để lưu mô hình trung gian
        class SaveModelCallback:
            def __init__(self, model, output_dir, save_intermediate):
                self.model = model
                self.output_dir = output_dir
                self.save_intermediate = save_intermediate
                self.best_score = -1
                self.best_epoch = -1

            def __call__(self, score, epoch, steps):
                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = epoch

                    # Lưu mô hình tốt nhất
                    if self.output_dir:
                        model_path = os.path.join(self.output_dir, "best_model")
                        self.model.save(model_path)
                        logger.info(f"Lưu mô hình tốt nhất tại epoch {epoch} với score {score:.4f}")

                # Lưu mô hình trung gian nếu được yêu cầu
                if self.save_intermediate and self.output_dir:
                    model_path = os.path.join(self.output_dir, f"model_epoch_{epoch}")
                    self.model.save(model_path)

        callback = SaveModelCallback(model, output_dir, save_intermediate)

        # Huấn luyện
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            optimizer_params={'lr': learning_rate},
            evaluation_steps=200,
            warmup_steps=100,
            output_path=output_dir,
            callback=callback,
            show_progress_bar=True
        )

        # Lưu mô hình cuối cùng
        if output_dir:
            model_path = os.path.join(output_dir, "final_model")
            model.save(model_path)

            logger.info(f"Đã lưu mô hình cuối cùng vào {model_path}")
            logger.info(f"Mô hình tốt nhất ở epoch {callback.best_epoch} với score {callback.best_score:.4f}")

            # Lưu metadata
            metadata = {
                "base_model": model_name,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_examples": len(examples),
                "best_epoch": callback.best_epoch,
                "best_score": callback.best_score
            }

            metadata_path = os.path.join(output_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("Huấn luyện hoàn tất")

    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình embedding: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


def main() -> None:
    """
    Hàm main để chạy huấn luyện mô hình embedding.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện hoặc tinh chỉnh mô hình embedding")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Thư mục chứa dữ liệu huấn luyện (các file PDF)"
    )

    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="File JSON chứa các cặp câu hỏi-đáp"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./trained_embeddings",
        help="Thư mục lưu mô hình đã huấn luyện"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Tên mô hình cơ sở"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Số epochs huấn luyện"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Kích thước batch"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Tốc độ học"
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Lưu mô hình sau mỗi epoch"
    )

    args = parser.parse_args()

    # Huấn luyện mô hình
    train_embedding_model(
        data_dir=args.data,
        query_pairs_file=args.queries,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_intermediate=args.save_intermediate
    )


if __name__ == "__main__":
    main()
