"""
Module mô hình ngôn ngữ lớn (LLM) cho việc sinh câu trả lời.
"""
import logging
from typing import Dict, List, Any, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate

from app.config import GROQ_API_KEY, LLM_MODEL, TEMPERATURE, SYSTEM_PROMPT, REFLECTION_PROMPT

logger = logging.getLogger(__name__)


class LLMModel:
    """
    Lớp chứa mô hình ngôn ngữ lớn (LLM) để sinh câu trả lời.
    """

    def __init__(self, model_name: str = LLM_MODEL, temperature: float = TEMPERATURE):
        """
        Khởi tạo mô hình LLM.

        Args:
            model_name (str, optional): Tên mô hình LLM. Mặc định là LLM_MODEL.
            temperature (float, optional): Nhiệt độ cho việc sinh câu trả lời. Mặc định là TEMPERATURE.
        """
        logger.info(f"Khởi tạo mô hình LLM: {model_name}")

        try:
            self.llm = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=model_name,
                temperature=temperature,
                max_tokens=4096,
                top_p=0.85,
            )
            logger.info(f"Đã khởi tạo mô hình LLM thành công: {model_name}")
        except Exception as e:
            logger.error(f"Lỗi khi khởi tạo mô hình LLM: {str(e)}")
            raise

    def _format_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        Định dạng ngữ cảnh từ các tài liệu đã truy xuất.

        Args:
            context_docs (List[Dict[str, Any]]): Danh sách các tài liệu

        Returns:
            str: Ngữ cảnh đã được định dạng
        """
        formatted_context = ""

        for i, doc in enumerate(context_docs, 1):
            text = doc.get("text", doc.get("page_content", ""))
            source = doc.get("metadata", {}).get("source", "Unknown")

            formatted_context += f"[Tài liệu {i}] {text}\n"
            formatted_context += f"Nguồn: {source}\n\n"

        return formatted_context.strip()

    def generate_response(self,
                          question: str,
                          context_docs: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Sinh câu trả lời dựa trên câu hỏi và ngữ cảnh (nếu có).

        Args:
            question (str): Câu hỏi của người dùng
            context_docs (Optional[List[Dict[str, Any]]], optional): Danh sách tài liệu ngữ cảnh. Mặc định là None.

        Returns:
            str: Câu trả lời từ mô hình
        """
        try:
            # Nếu có ngữ cảnh, sử dụng để trả lời
            if context_docs:
                context = self._format_context(context_docs)

                prompt = PromptTemplate(
                    template=SYSTEM_PROMPT,
                    input_variables=["context", "question"]
                )

                chain = prompt | self.llm

                result = chain.invoke({"context": context, "question": question})
                return result.content

            # Nếu không có ngữ cảnh, trả lời dựa trên kiến thức sẵn có
            else:
                messages = [
                    SystemMessage(
                        content="Bạn là trợ lý tư vấn tuyển sinh đại học. Hãy trả lời câu hỏi của học sinh một cách ngắn gọn và hữu ích."),
                    HumanMessage(content=question)
                ]

                result = self.llm.invoke(messages)
                return result.content

        except Exception as e:
            logger.error(f"Lỗi khi sinh câu trả lời: {str(e)}")
            error_message = "Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn. Vui lòng thử lại sau."
            return error_message

    def generate_reflection(self, question: str, reflection_prompt: str = REFLECTION_PROMPT) -> str:
        """
        Phân tích câu hỏi để xác định cách xử lý.

        Args:
            question (str): Câu hỏi của người dùng
            reflection_prompt (str): Prompt cho việc phân tích. Mặc định là REFLECTION_PROMPT.

        Returns:
            str: Kết quả phân tích ("RAG" hoặc "LLM")
        """
        try:
            prompt = PromptTemplate(
                template=reflection_prompt,
                input_variables=["question"]
            )

            chain = prompt | self.llm

            result = chain.invoke({"question": question})
            return result.content.strip()

        except Exception as e:
            logger.error(f"Lỗi khi phân tích câu hỏi: {str(e)}")
            # Mặc định sử dụng RAG nếu có lỗi
            return "RAG"


# Singleton để giảm chi phí khởi tạo mô hình nhiều lần
_llm_model_instance = None


def get_llm_model() -> LLMModel:
    """
    Trả về instance singleton của mô hình LLM.

    Returns:
        LLMModel: Instance mô hình LLM
    """
    global _llm_model_instance

    if _llm_model_instance is None:
        _llm_model_instance = LLMModel()

    return _llm_model_instance