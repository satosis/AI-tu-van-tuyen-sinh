"""
Module phân tích và phản ánh câu hỏi của người dùng.
"""
import logging
import re
import json
from typing import Dict, Any

from app.models.llm import get_llm_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

# Prompt cho việc phân tích câu hỏi chi tiết
DETAILED_REFLECTION_PROMPT = """
Hãy phân tích câu hỏi sau đây về thông tin tuyển sinh:

Câu hỏi: {question}

Vui lòng trả lời các câu hỏi sau:
1. Câu hỏi này có phải là về tuyển sinh không? (Có/Không)
2. Chủ đề chính của câu hỏi là gì? (Ví dụ: điểm chuẩn, ngành học, học phí, v.v.)
3. Liệt kê các từ khóa quan trọng trong câu hỏi.
4. Câu hỏi này cần thông tin cụ thể từ tài liệu hay chỉ là câu hỏi chung chung?
5. Có thể diễn đạt lại câu hỏi này để dễ tìm kiếm thông tin trong tài liệu không?

Trả lời theo định dạng JSON:
{{
    "is_admission_related": true/false,
    "main_topic": "<chủ đề>",
    "keywords": ["từ khóa 1", "từ khóa 2", ...],
    "needs_specific_info": true/false,
    "rephrased_question": "<câu hỏi được diễn đạt lại>"
}}
"""


class Reflection:
    """
    Lớp phân tích và phản ánh câu hỏi.
    """

    def __init__(self):
        """
        Khởi tạo reflection model.
        """
        logger.info("Khởi tạo reflection model")

        try:
            # Lấy instance LLM
            self.llm = get_llm_model()

            logger.info("Đã khởi tạo reflection model thành công")

        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo reflection model: {str(e)}")
            raise

    def simple_analysis(self, query: str) -> Dict[str, Any]:
        """
        Phân tích cơ bản câu hỏi của người dùng.

        Args:
            query (str): Câu hỏi cần phân tích

        Returns:
            Dict[str, Any]: Kết quả phân tích cơ bản
        """
        try:
            # Độ dài
            length = len(query)

            # Số từ (ước tính)
            words = len(re.findall(r'\w+', query))

            # Có dấu hỏi không
            has_question_mark = '?' in query

            # Từ khóa liên quan đến tuyển sinh
            admission_keywords = [
                'tuyển sinh', 'ngành', 'điểm', 'điểm chuẩn', 'học phí', 'học bổng',
                'đại học', 'cao đẳng', 'đào tạo', 'trường', 'khoa', 'ngành học',
                'chuyên ngành', 'thi', 'xét tuyển', 'hồ sơ', 'nộp', 'đơn', 'đăng ký',
                'thư viện', 'cơ sở vật chất', 'phòng'
            ]

            found_keywords = [keyword for keyword in admission_keywords if keyword.lower() in query.lower()]

            # Có liên quan đến tuyển sinh không
            is_admission_related = len(found_keywords) > 0

            result = {
                "length": length,
                "word_count": words,
                "has_question_mark": has_question_mark,
                "found_keywords": found_keywords,
                "is_admission_related": is_admission_related
            }

            logger.info(f"Đã phân tích cơ bản câu hỏi: {query}")
            return result

        except Exception as e:
            logger.error(f"Lỗi khi phân tích cơ bản câu hỏi: {str(e)}")
            return {"error": str(e)}

    def detailed_analysis(self, query: str) -> Dict[str, Any]:
        """
        Phân tích chi tiết câu hỏi của người dùng bằng LLM.

        Args:
            query (str): Câu hỏi cần phân tích

        Returns:
            Dict[str, Any]: Kết quả phân tích chi tiết
        """
        try:
            # Phân tích cơ bản
            basic_analysis = self.simple_analysis(query)

            # Nếu có vẻ không liên quan đến tuyển sinh, trả về phân tích cơ bản
            if not basic_analysis.get("is_admission_related", False) and not basic_analysis.get("error"):
                logger.info(f"Câu hỏi không liên quan đến tuyển sinh, trả về phân tích cơ bản")
                return basic_analysis

            # Nếu liên quan đến tuyển sinh, phân tích chi tiết bằng LLM
            prompt = PromptTemplate(
                template=DETAILED_REFLECTION_PROMPT,
                input_variables=["question"]
            )

            chain = LLMChain(llm=self.llm.llm, prompt=prompt)

            result = chain.invoke({"question": query})
            result_text = result["text"]

            # Trích xuất phần JSON từ kết quả
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Nếu không tìm thấy định dạng ```json```, thử tìm trực tiếp đối tượng JSON
                json_match = re.search(r'\{\s*"is_admission_related".*\}', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    logger.warning(f"Không tìm thấy phần JSON trong kết quả LLM: {result_text}")
                    return basic_analysis

            # Parse JSON
            try:
                detailed_result = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi khi parse JSON: {str(e)}")
                logger.error(f"JSON string: {json_str}")
                return basic_analysis

            # Gộp kết quả
            combined_result = {**basic_analysis, **detailed_result}

            logger.info(f"Đã phân tích chi tiết câu hỏi: {query}")
            return combined_result

        except Exception as e:
            logger.error(f"Lỗi khi phân tích chi tiết câu hỏi: {str(e)}")
            return basic_analysis if 'basic_analysis' in locals() else {"error": str(e)}

    def rephrase_query(self, query: str) -> str:
        """
        Diễn đạt lại câu hỏi để tối ưu cho việc tìm kiếm.

        Args:
            query (str): Câu hỏi gốc

        Returns:
            str: Câu hỏi được diễn đạt lại
        """
        try:
            # Phân tích chi tiết
            analysis = self.detailed_analysis(query)

            # Nếu đã có câu hỏi được diễn đạt lại, trả về
            if "rephrased_question" in analysis and analysis["rephrased_question"]:
                rephrased = analysis["rephrased_question"]
                logger.info(f"Đã diễn đạt lại câu hỏi: '{query}' -> '{rephrased}'")
                return rephrased

            # Nếu không, trả về câu hỏi gốc
            return query

        except Exception as e:
            logger.error(f"Lỗi khi diễn đạt lại câu hỏi: {str(e)}")
            return query


# Singleton để giảm chi phí khởi tạo nhiều lần
_reflection_instance = None


def get_reflection() -> Reflection:
    """
    Trả về instance singleton của reflection model.

    Returns:
        Reflection: Instance reflection model
    """
    global _reflection_instance

    if _reflection_instance is None:
        _reflection_instance = Reflection()

    return _reflection_instance