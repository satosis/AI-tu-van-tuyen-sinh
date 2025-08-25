"""
Cấu hình cho ứng dụng chatbot tư vấn tuyển sinh.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM settings
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Database settings
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

# API settings
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Chunk settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# RAG settings
TOP_K = 5  # Số lượng chunks lấy ra từ vector database
TEMPERATURE = 0.1  # Độ sáng tạo của LLM

# System prompts
SYSTEM_PROMPT = """
Bạn là trợ lý tư vấn tuyển sinh đại học, nhiệm vụ của bạn là cung cấp thông tin chính xác về tuyển sinh cho học sinh.
Hãy trả lời dựa trên thông tin được cung cấp bên dưới.
Nếu không có thông tin hoặc thông tin không rõ ràng, hãy thừa nhận và đề nghị học sinh liên hệ trực tiếp với nhà trường.
Hãy trả lời ngắn gọn, súc tích và dễ hiểu.

{context}

Câu hỏi: {question}
"""

REFLECTION_PROMPT = """
Hãy phân tích câu hỏi sau đây và xác định liệu nó có liên quan đến thông tin tuyển sinh không:

Câu hỏi: {question}

Nếu câu hỏi liên quan đến thông tin tuyển sinh (như điểm chuẩn, ngành học, chương trình đào tạo, học phí, học bổng, ký túc xá, quy trình nhập học),
hãy trả về "RAG". Nếu câu hỏi là chào hỏi thông thường hoặc không liên quan đến thông tin tụ vấn tuyển sinh, hãy trả về "LLM".

Trả lời chỉ "RAG" hoặc "LLM":
"""

# Evaluation settings
EVAL_METRICS = ["relevance", "completeness", "consistency"]