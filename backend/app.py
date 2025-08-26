"""
Backend Flask API cho chatbot tư vấn tuyển sinh.
"""
import os
import logging
import traceback
from typing import Dict, Any, List

from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
except ModuleNotFoundError:
    class CORS:  # pragma: no cover - used when flask-cors isn't installed
        def __init__(self, app: Flask | None = None, *args, **kwargs):
            if app is not None:
                self.init_app(app)

        def init_app(self, app: Flask) -> Flask:
            logging.getLogger(__name__).warning(
                "`flask-cors` is not installed; CORS support disabled.")
            return app

# Thêm đường dẫn gốc vào sys.path
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Đảm bảo luồng stdout/stderr sử dụng UTF-8 để tránh lỗi UnicodeEncodeError trên Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from app.config import BACKEND_HOST, BACKEND_PORT
from app.pipeline.router import get_semantic_router
from app.pipeline.reflection import get_reflection
from app.vectordb.faiss_client import get_faiss_client
from app.models.llm import get_llm_model
from app.models.evaluation import get_evaluation_model

# Cấu hình logging nếu chưa có handler nào
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('backend.log', encoding='utf-8')
        ]
    )

logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
if CORS:
    CORS(app)  # Cho phép CORS khi thư viện có sẵn

# Khởi tạo các thành phần cần thiết
# Các thành phần sẽ được lazy loading khi gọi API
router = None
reflection = None
faiss_client = None
llm = None
evaluator = None


def lazy_init() -> None:
    """
    Khởi tạo lười các thành phần cần thiết.
    """
    global router, reflection, faiss_client, llm, evaluator

    try:
        if router is None:
            router = get_semantic_router()

        if reflection is None:
            reflection = get_reflection()

        if faiss_client is None:
            faiss_client = get_faiss_client()

        if llm is None:
            llm = get_llm_model()

        if evaluator is None:
            evaluator = get_evaluation_model()

    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo các thành phần: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.route('/api/health', methods=['GET'])
def health_check() -> Dict[str, Any]:
    """
    Kiểm tra trạng thái của API.

    Returns:
        Dict[str, Any]: Thông tin trạng thái
    """
    return jsonify({
        "status": "ok",
        "message": "API đang hoạt động bình thường"
    })


@app.route('/api/chat', methods=['POST'])
def chat() -> Dict[str, Any]:
    """
    Xử lý câu hỏi của người dùng.

    Returns:
        Dict[str, Any]: Kết quả xử lý
    """
    try:
        # Lấy dữ liệu từ request
        data = request.json

        if not data or 'query' not in data:
            return jsonify({
                "error": "Dữ liệu không hợp lệ. Cần có trường 'query'."
            }), 400

        query = data.get('query', '').strip()
        evaluate = data.get('evaluate', False)

        if not query:
            return jsonify({
                "error": "Câu hỏi không được để trống."
            }), 400

        # Khởi tạo lười các thành phần
        lazy_init()

        # Xử lý câu hỏi
        logger.info(f"Xử lý câu hỏi: {query}")

        # Phân tích và định tuyến câu hỏi
        result = router.route_query(query, evaluate=evaluate)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý câu hỏi: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            "error": "Đã xảy ra lỗi khi xử lý câu hỏi. Vui lòng thử lại sau.",
            "details": str(e)
        }), 500


@app.route('/api/collection/stats', methods=['GET'])
def get_index_stats() -> Dict[str, Any]:
    """
    Lấy thông tin thống kê của FAISS index.

    Returns:
        Dict[str, Any]: Thông tin thống kê
    """
    try:
        # Khởi tạo lười các thành phần
        lazy_init()

        # Lấy thông tin thống kê
        stats = faiss_client.get_collection_stats()

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin index: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            "error": "Đã xảy ra lỗi khi lấy thông tin index.",
            "details": str(e)
        }), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_response() -> Dict[str, Any]:
    """
    Đánh giá câu trả lời.

    Returns:
        Dict[str, Any]: Kết quả đánh giá
    """
    try:
        # Lấy dữ liệu từ request
        data = request.json

        if not data or 'question' not in data or 'answer' not in data or 'context' not in data:
            return jsonify({
                "error": "Dữ liệu không hợp lệ. Cần có các trường 'question', 'answer', và 'context'."
            }), 400

        question = data.get('question', '').strip()
        answer = data.get('answer', '').strip()
        context = data.get('context', '').strip()

        if not question or not answer:
            return jsonify({
                "error": "Câu hỏi và câu trả lời không được để trống."
            }), 400

        # Khởi tạo lười các thành phần
        lazy_init()

        # Đánh giá câu trả lời
        logger.info(f"Đánh giá câu trả lời cho câu hỏi: {question}")

        evaluation = evaluator.evaluate_response(question, answer, context)

        return jsonify(evaluation)

    except Exception as e:
        logger.error(f"Lỗi khi đánh giá câu trả lời: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            "error": "Đã xảy ra lỗi khi đánh giá câu trả lời.",
            "details": str(e)
        }), 500


@app.route('/api/reflection', methods=['POST'])
def analyze_query() -> Dict[str, Any]:
    """
    Phân tích câu hỏi.

    Returns:
        Dict[str, Any]: Kết quả phân tích
    """
    try:
        # Lấy dữ liệu từ request
        data = request.json

        if not data or 'query' not in data:
            return jsonify({
                "error": "Dữ liệu không hợp lệ. Cần có trường 'query'."
            }), 400

        query = data.get('query', '').strip()
        detailed = data.get('detailed', False)

        if not query:
            return jsonify({
                "error": "Câu hỏi không được để trống."
            }), 400

        # Khởi tạo lười các thành phần
        lazy_init()

        # Phân tích câu hỏi
        logger.info(f"Phân tích câu hỏi: {query}")

        if detailed:
            analysis = reflection.detailed_analysis(query)
        else:
            analysis = reflection.simple_analysis(query)

        return jsonify(analysis)

    except Exception as e:
        logger.error(f"Lỗi khi phân tích câu hỏi: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            "error": "Đã xảy ra lỗi khi phân tích câu hỏi.",
            "details": str(e)
        }), 500


def main() -> None:
    """
    Hàm main để chạy ứng dụng Flask.
    """
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=True)


if __name__ == "__main__":
    main()