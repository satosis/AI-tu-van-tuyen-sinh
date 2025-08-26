"""
Script để đánh giá hiệu suất của chatbot tư vấn tuyển sinh.
"""
import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc vào sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Cấu hình logging trước khi import các module có thể sử dụng logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)

logger = logging.getLogger(__name__)

from app.config import EVAL_METRICS
from app.pipeline.router import get_semantic_router
from app.models.evaluation import get_evaluation_model


def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    """
    Tải danh sách câu hỏi kiểm thử từ file.

    Args:
        file_path (str): Đường dẫn đến file câu hỏi kiểm thử

    Returns:
        List[Dict[str, Any]]: Danh sách câu hỏi kiểm thử
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        logger.info(f"Đã tải {len(questions)} câu hỏi kiểm thử từ {file_path}")
        return questions

    except Exception as e:
        logger.error(f"Lỗi khi tải câu hỏi kiểm thử: {str(e)}")
        return []


def run_evaluation(questions: List[Dict[str, Any]],
                   output_dir: str = "./evaluation_results",
                   save_results: bool = True) -> Dict[str, Any]:
    """
    Chạy đánh giá trên danh sách câu hỏi.

    Args:
        questions (List[Dict[str, Any]]): Danh sách câu hỏi kiểm thử
        output_dir (str, optional): Thư mục lưu kết quả đánh giá. Mặc định là "./evaluation_results".
        save_results (bool, optional): Có lưu kết quả không. Mặc định là True.

    Returns:
        Dict[str, Any]: Kết quả đánh giá tổng hợp
    """
    try:
        # Khởi tạo router và evaluator
        router = get_semantic_router()
        evaluator = get_evaluation_model()

        results = []

        # Xử lý từng câu hỏi
        for i, question_data in enumerate(questions, 1):
            question = question_data.get("question", "")
            expected_answer = question_data.get("expected_answer", "")

            if not question:
                logger.warning(f"Bỏ qua câu hỏi {i} do không có nội dung")
                continue

            logger.info(f"Đang xử lý câu hỏi {i}/{len(questions)}: {question}")

            # Xử lý câu hỏi bằng router
            response = router.route_query(question, evaluate=True)

            # Lưu kết quả
            result = {
                "question": question,
                "route": response.get("route", ""),
                "answer": response.get("answer", ""),
                "expected_answer": expected_answer,
                "source": response.get("source", ""),
                "evaluation": response.get("evaluation", {}),
            }

            results.append(result)

            logger.info(f"Đã xử lý xong câu hỏi {i}")

        # Tính toán số liệu thống kê
        stats = calculate_statistics(results)

        # Lưu kết quả nếu được yêu cầu
        if save_results:
            # Đảm bảo thư mục tồn tại
            os.makedirs(output_dir, exist_ok=True)

            # Lưu kết quả chi tiết
            detailed_output_path = os.path.join(output_dir, "detailed_results.json")
            with open(detailed_output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            # Lưu số liệu thống kê
            stats_output_path = os.path.join(output_dir, "statistics.json")
            with open(stats_output_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            # Tạo báo cáo đánh giá
            create_evaluation_report(results, stats, output_dir)

            logger.info(f"Đã lưu kết quả đánh giá vào {output_dir}")

        return {
            "detailed_results": results,
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Lỗi khi chạy đánh giá: {str(e)}")
        return {"error": str(e)}


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tính toán số liệu thống kê từ kết quả đánh giá.

    Args:
        results (List[Dict[str, Any]]): Danh sách kết quả đánh giá

    Returns:
        Dict[str, Any]: Số liệu thống kê
    """
    # Tổng số câu hỏi
    total_questions = len(results)

    if total_questions == 0:
        return {"error": "Không có kết quả để tính toán thống kê"}

    # Phân loại theo route
    rag_count = sum(1 for result in results if result.get("route") == "RAG")
    llm_count = sum(1 for result in results if result.get("route") == "LLM")

    # Tính điểm trung bình cho các metrics
    metrics_data = {}
    for metric in EVAL_METRICS + ["overall"]:
        values = []
        for result in results:
            eval_data = result.get("evaluation", {})
            if metric == "overall":
                if "overall" in eval_data:
                    values.append(eval_data["overall"])
            elif metric in eval_data and "score" in eval_data[metric]:
                values.append(eval_data[metric]["score"])

        if values:
            metrics_data[metric] = {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }

    # Thống kê tổng hợp
    stats = {
        "total_questions": total_questions,
        "route_distribution": {
            "RAG": rag_count,
            "LLM": llm_count,
            "RAG_percentage": (rag_count / total_questions) * 100 if total_questions > 0 else 0,
            "LLM_percentage": (llm_count / total_questions) * 100 if total_questions > 0 else 0
        },
        "metrics": metrics_data
    }

    return stats


def create_evaluation_report(results: List[Dict[str, Any]],
                             stats: Dict[str, Any],
                             output_dir: str) -> None:
    """
    Tạo báo cáo đánh giá với các biểu đồ và bảng.

    Args:
        results (List[Dict[str, Any]]): Danh sách kết quả đánh giá
        stats (Dict[str, Any]): Số liệu thống kê
        output_dir (str): Thư mục lưu báo cáo
    """
    # 1. Biểu đồ phân bố route
    route_dist = stats.get("route_distribution", {})

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["RAG", "LLM"]
    values = [route_dist.get("RAG", 0), route_dist.get("LLM", 0)]

    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4285F4', '#34A853'])
    ax.axis('equal')
    plt.title('Phân bố RAG vs LLM')
    plt.tight_layout()

    route_chart_path = os.path.join(output_dir, "route_distribution.png")
    plt.savefig(route_chart_path)
    plt.close()

    # 2. Biểu đồ điểm trung bình các metrics
    metrics_data = stats.get("metrics", {})

    if metrics_data:
        metrics = list(metrics_data.keys())
        mean_scores = [metrics_data[m].get("mean", 0) for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(metrics, mean_scores, color='#4285F4')
        ax.set_ylim(0, 10)
        ax.set_ylabel('Điểm trung bình')
        ax.set_title('Điểm trung bình các metrics')

        for i, v in enumerate(mean_scores):
            ax.text(i, v + 0.1, f"{v:.2f}", ha='center')

        plt.tight_layout()

        metrics_chart_path = os.path.join(output_dir, "metrics_scores.png")
        plt.savefig(metrics_chart_path)
        plt.close()

    # 3. Tạo file CSV cho kết quả chi tiết
    data_rows = []

    for result in results:
        eval_data = result.get("evaluation", {})

        row = {
            "question": result.get("question", ""),
            "route": result.get("route", ""),
            "source": result.get("source", ""),
        }

        # Thêm điểm đánh giá nếu có
        for metric in EVAL_METRICS:
            if metric in eval_data and "score" in eval_data[metric]:
                row[f"{metric}_score"] = eval_data[metric]["score"]

        if "overall" in eval_data:
            row["overall_score"] = eval_data["overall"]

        data_rows.append(row)

    if data_rows:
        df = pd.DataFrame(data_rows)
        csv_path = os.path.join(output_dir, "evaluation_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')


def main() -> None:
    """
    Hàm main để chạy đánh giá.
    """
    parser = argparse.ArgumentParser(description="Đánh giá hiệu suất của chatbot tư vấn tuyển sinh")

    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Đường dẫn đến file JSON chứa danh sách câu hỏi kiểm thử"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="Thư mục lưu kết quả đánh giá"
    )

    args = parser.parse_args()

    # Tải câu hỏi kiểm thử
    questions = load_test_questions(args.questions)

    if not questions:
        logger.error("Không có câu hỏi kiểm thử để đánh giá")
        return

    # Chạy đánh giá
    run_evaluation(questions, args.output)


if __name__ == "__main__":
    main()

