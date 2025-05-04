"""
Module đánh giá chất lượng câu trả lời của chatbot.
"""
import logging
from typing import Dict, List, Any

from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from app.config import GROQ_API_KEY, LLM_MODEL, EVAL_METRICS
from app.models.embedding import get_embedding_model

logger = logging.getLogger(__name__)

# Prompt để đánh giá chất lượng câu trả lời
EVAL_PROMPT = """
Bạn là hệ thống đánh giá câu trả lời từ chatbot tư vấn tuyển sinh. 
Nhiệm vụ của bạn là đánh giá câu trả lời dựa trên những tiêu chí sau:

1. Độ liên quan (Relevance): Câu trả lời có đáp ứng đúng câu hỏi không?
2. Tính đầy đủ (Completeness): Câu trả lời có đầy đủ thông tin cần thiết không?
3. Tính nhất quán (Consistency): Câu trả lời có nhất quán với thông tin nguồn không?

Câu hỏi: {question}
Ngữ cảnh: {context}
Câu trả lời: {answer}

Hãy đánh giá câu trả lời trên thang điểm từ 1-10 cho mỗi tiêu chí và giải thích lý do.
Trả về định dạng JSON như sau:
{{
    "relevance": {{
        "score": <điểm từ 1-10>,
        "explanation": "<lý do>"
    }},
    "completeness": {{
        "score": <điểm từ 1-10>,
        "explanation": "<lý do>"
    }},
    "consistency": {{
        "score": <điểm từ 1-10>,
        "explanation": "<lý do>"
    }},
    "overall": <điểm trung bình từ 1-10>,
    "summary": "<tóm tắt đánh giá>"
}}
"""


class EvaluationModel:
    """
    Lớp đánh giá chất lượng câu trả lời của chatbot.
    """

    def __init__(self, model_name: str = LLM_MODEL):
        """
        Khởi tạo mô hình đánh giá.

        Args:
            model_name (str, optional): Tên mô hình LLM. Mặc định là LLM_MODEL.
        """
        logger.info(f"Khởi tạo mô hình đánh giá: {model_name}")

        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=model_name,
                temperature=0.1,  # Nhiệt độ thấp cho đánh giá khách quan
                max_tokens=2048,
            )
            logger.info(f"Đã khởi tạo mô hình đánh giá thành công: {model_name}")

            # Embedding model để tính độ tương đồng
            self.embedding = get_embedding_model()

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình đánh giá: {str(e)}")
            raise

    def _calculate_semantic_similarity(self, question: str, answer: str) -> float:
        """
        Tính độ tương đồng ngữ nghĩa giữa câu hỏi và câu trả lời.

        Args:
            question (str): Câu hỏi
            answer (str): Câu trả lời

        Returns:
            float: Độ tương đồng (từ 0 đến 1)
        """
        try:
            return self.embedding.calculate_similarity(question, answer)
        except Exception as e:
            logger.error(f"Lỗi khi tính độ tương đồng: {str(e)}")
            return 0.0

    def evaluate_response(self,
                          question: str,
                          answer: str,
                          context: str) -> Dict[str, Any]:
        """
        Đánh giá chất lượng câu trả lời.

        Args:
            question (str): Câu hỏi của người dùng
            answer (str): Câu trả lời của chatbot
            context (str): Ngữ cảnh (thông tin tham khảo)

        Returns:
            Dict[str, Any]: Kết quả đánh giá
        """
        try:
            # Tính toán độ tương đồng ngữ nghĩa
            similarity = self._calculate_semantic_similarity(question, answer)

            # Đánh giá bằng LLM
            prompt = PromptTemplate(
                template=EVAL_PROMPT,
                input_variables=["question", "context", "answer"]
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            result = chain.invoke({
                "question": question,
                "context": context,
                "answer": answer
            })

            # Parse kết quả JSON từ LLM
            import json
            eval_result = {}

            try:
                eval_result = json.loads(result["text"])
                # Thêm độ tương đồng ngữ nghĩa vào kết quả
                eval_result["semantic_similarity"] = similarity

            except json.JSONDecodeError:
                logger.error("Không thể parse kết quả đánh giá JSON")
                eval_result = {
                    "relevance": {"score": 5, "explanation": "Không thể đánh giá"},
                    "completeness": {"score": 5, "explanation": "Không thể đánh giá"},
                    "consistency": {"score": 5, "explanation": "Không thể đánh giá"},
                    "overall": 5,
                    "summary": "Không thể đánh giá chính xác do lỗi định dạng",
                    "semantic_similarity": similarity
                }

            return eval_result

        except Exception as e:
            logger.error(f"Lỗi khi đánh giá câu trả lời: {str(e)}")
            return {
                "error": str(e),
                "overall": 0
            }


# Singleton để giảm chi phí khởi tạo mô hình nhiều lần
_evaluation_model_instance = None


def get_evaluation_model() -> EvaluationModel:
    """
    Trả về instance singleton của mô hình đánh giá.

    Returns:
        EvaluationModel: Instance mô hình đánh giá
    """
    global _evaluation_model_instance

    if _evaluation_model_instance is None:
        _evaluation_model_instance = EvaluationModel()

    return _evaluation_model_instance