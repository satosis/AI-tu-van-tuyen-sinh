"""
Định nghĩa các routes cho backend Flask.
"""
from flask import Blueprint, request, jsonify
import logging
import traceback
from typing import Dict, Any

from app.pipeline.router import get_semantic_router
from app.pipeline.reflection import get_reflection
from app.vectordb.chroma_client import get_chroma_client
from app.models.evaluation import get_evaluation_model

logger = logging.getLogger(__name__)

# Tạo blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Lazy loading các thành phần
router = None
reflection = None
chroma_client = None
evaluator = None


def lazy_init() -> None:
    """
    Khởi tạo lười các thành phần cần thiết.
    """
    global router, reflection, chroma_client, evaluator

    try:
        if router is None:
            router = get_semantic_router()

        if reflection is None:
            reflection = get_reflection()

        if chroma_client is None:
            chroma_client = get_chroma_client()

        if evaluator is None:
            evaluator = get_evaluation_model()

    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo các thành phần: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@api_bp.route('/health', methods=['GET'])
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


@api_bp.route('/chat', methods=['POST'])
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


@api_bp.route('/collection/stats', methods=['GET'])
def get_collection_stats() -> Dict[str, Any]:
    """
    Lấy thông tin thống kê của collection.

    Returns:
        Dict[str, Any]: Thông tin thống kê
    """
    try:
        # Khởi tạo lười các thành phần
        lazy_init()

        # Lấy thông tin thống kê
        stats = chroma_client.get_collection_stats()

        return jsonify(stats)

    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin collection: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            "error": "Đã xảy ra lỗi khi lấy thông tin collection.",
            "details": str(e)
        }), 500


@api_bp.route('/evaluate', methods=['POST'])
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


@api_bp.route('/reflection', methods=['POST'])
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


@api_bp.route('/search', methods=['POST'])
def search_documents() -> Dict[str, Any]:
    """
    Tìm kiếm tài liệu.

    Returns:
        Dict[str, Any]: Kết quả tìm kiếm
    """
    try:
        # Lấy dữ liệu từ request
        data = request.json

        if not data or 'query' not in data:
            return jsonify({
                "error": "Dữ liệu không hợp lệ. Cần có trường 'query'."
            }), 400

        query = data.get('query', '').strip()
        n_results = data.get('n_results', 5)

        if not query:
            return jsonify({
                "error": "Truy vấn tìm kiếm không được để trống."
            }), 400

        # Khởi tạo lười các thành phần
        lazy_init()

        # Tìm kiếm tài liệu
        logger.info(f"Tìm kiếm tài liệu với truy vấn: {query}")

        from app.vectordb.hybrid_search import get_hybrid_search
        hybrid_search = get_hybrid_search()

        results = hybrid_search.search(query, n_results=n_results)

        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })

    except Exception as e:
        logger.error(f"Lỗi khi tìm kiếm tài liệu: {str(e)}")
        logger.error(traceback.format_exc())

        return jsonify({
            "error": "Đã xảy ra lỗi khi tìm kiếm tài liệu.",
            "details": str(e)
        }), 500