"""
Giao diện Streamlit cho chatbot tư vấn tuyển sinh.
"""
import os
import json
import requests
import time
from datetime import datetime
import logging
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

# Thêm đường dẫn gốc vào sys.path - FIX IMPORT ERROR
import sys
from pathlib import Path
# Đảm bảo thư mục gốc được thêm vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Sau khi đã thêm đường dẫn gốc, import config
from app.config import BACKEND_HOST, BACKEND_PORT, FRONTEND_PORT

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('frontend.log', encoding='utf-8')  # Thêm encoding='utf-8' để tránh lỗi Unicode
    ]
)

logger = logging.getLogger(__name__)

# URL API Backend
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api"

# Cấu hình trang
st.set_page_config(
    page_title="Chatbot Tư Vấn Tuyển Sinh",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Định nghĩa CSS
st.markdown("""
<style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    * {
        font-family: 'Roboto', sans-serif;
    }

    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2979FF;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #E0E0E0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Sub header */
    .sub-header {
        font-size: 1.5rem;
        color: #616161;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Chat container */
    .chat-container {
        max-width: 90%;
        margin: 0 auto;
    }

    /* Chat message styles */
    .chat-message {
        display: flex;
        margin-bottom: 1.5rem;
    }

    .user-message {
        justify-content: flex-end;
    }

    .bot-message {
        justify-content: flex-start;
    }

    .message-bubble {
        padding: 1rem 1.5rem;
        border-radius: 18px;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
    }

    .user-bubble {
        background: linear-gradient(135deg, #2979FF, #1565C0);
        color: white;
        border-bottom-right-radius: 5px;
    }

    .bot-bubble {
        background: #F5F5F5;
        color: #212121;
        border-bottom-left-radius: 5px;
        border-left: 3px solid #2979FF;
    }

    .message-content {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }

    .user-bubble .message-content {
        color: white;
    }

    .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .bot-bubble .message-header {
        color: #2979FF;
    }

    .user-bubble .message-header {
        color: rgba(255, 255, 255, 0.9);
    }

    .message-time {
        font-size: 0.75rem;
        opacity: 0.8;
    }

    .source-info {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 0.8rem;
        border-top: 1px solid #EEEEEE;
        padding-top: 0.5rem;
        font-style: italic;
    }

    /* Evaluation metrics */
    .eval-metrics {
        margin-top: 1.5rem;
        padding: 1.2rem;
        background-color: #FAFAFA;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .eval-metrics h4 {
        color: #2979FF;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    .metric-item {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }

    .metric-name {
        font-weight: 500;
    }

    .metric-value {
        color: #2979FF;
        font-weight: 500;
    }

    /* Context box */
    .context-box {
        margin-top: 1.5rem;
        padding: 1.2rem;
        background-color: #F5F7FA;
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        max-height: 300px;
        overflow-y: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .context-box h4 {
        color: #2979FF;
        margin-bottom: 1rem;
        font-weight: 500;
    }

    /* Input area */
    .input-area {
        display: flex;
        margin-top: 2rem;
        padding: 1rem;
        background: #F5F7FA;
        border-radius: 12px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #FAFAFA;
        padding: 1rem;
    }

    .sidebar h1 {
        color: #2979FF;
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }

    .sidebar h3 {
        color: #424242;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #E0E0E0;
        padding-bottom: 0.5rem;
    }

    /* Button styling */
    .stButton > button {
        background-color: #2979FF;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #E0E0E0;
        padding: 0.8rem 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: #2979FF;
        box-shadow: 0 0 0 2px rgba(41, 121, 255, 0.2);
    }

    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #2979FF !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2979FF !important;
        color: white !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #424242;
        background-color: #F5F7FA;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #F5F5F5;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #BDBDBD;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #9E9E9E;
    }
</style>
""", unsafe_allow_html=True)

def check_backend_health() -> bool:
    """
    Kiểm tra trạng thái của backend.

    Returns:
        bool: True nếu backend hoạt động bình thường, False nếu không
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False

def send_chat_message(query: str, evaluate: bool = False) -> Dict[str, Any]:
    """
    Gửi câu hỏi đến backend và nhận câu trả lời.

    Args:
        query (str): Câu hỏi của người dùng
        evaluate (bool, optional): Có đánh giá câu trả lời không. Mặc định là False.

    Returns:
        Dict[str, Any]: Kết quả từ backend
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": query, "evaluate": evaluate},
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        logger.error(f"Lỗi khi gửi câu hỏi đến backend: {str(e)}")
        return {
            "error": "Không thể kết nối đến server. Vui lòng thử lại sau.",
            "details": str(e)
        }

def get_index_stats() -> Dict[str, Any]:
    """
    Lấy thông tin thống kê của FAISS index.

    Returns:
        Dict[str, Any]: Thông tin thống kê
    """
    try:
        response = requests.get(f"{BACKEND_URL}/collection/stats", timeout=5)
        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        logger.error(f"Lỗi khi lấy thông tin index: {str(e)}")
        return {"error": str(e)}

def analyze_query(query: str, detailed: bool = False) -> Dict[str, Any]:
    """
    Phân tích câu hỏi.

    Args:
        query (str): Câu hỏi cần phân tích
        detailed (bool, optional): Có phân tích chi tiết không. Mặc định là False.

    Returns:
        Dict[str, Any]: Kết quả phân tích
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/reflection",
            json={"query": query, "detailed": detailed},
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        logger.error(f"Lỗi khi phân tích câu hỏi: {str(e)}")
        return {"error": str(e)}

def search_documents(query: str, n_results: int = 5) -> Dict[str, Any]:
    """
    Tìm kiếm tài liệu.

    Args:
        query (str): Truy vấn tìm kiếm
        n_results (int, optional): Số lượng kết quả. Mặc định là 5.

    Returns:
        Dict[str, Any]: Kết quả tìm kiếm
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            json={"query": query, "n_results": n_results},
            timeout=10
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        logger.error(f"Lỗi khi tìm kiếm tài liệu: {str(e)}")
        return {"error": str(e)}

def render_sidebar() -> None:
    """
    Hiển thị sidebar với thông tin và tùy chọn.
    """
    st.sidebar.title("🎓 Thông tin chatbot")

    st.sidebar.markdown("""
    **Chatbot Tư Vấn Tuyển Sinh** là hệ thống hỗ trợ thông tin tuyển sinh dựa trên công nghệ RAG
    (Retrieval-Augmented Generation) và LLM.
    """)

    # Hiển thị thông tin hệ thống
    st.sidebar.subheader("Thông tin hệ thống")

    # Kiểm tra kết nối backend
    backend_status = "🟢 Hoạt động" if check_backend_health() else "🔴 Không hoạt động"
    st.sidebar.markdown(f"**Trạng thái backend:** {backend_status}")

    # Lấy thông tin index
    collection_stats = get_index_stats()

    if "error" not in collection_stats:
        st.sidebar.markdown(f"**Index:** {collection_stats.get('name', 'N/A')}")
        st.sidebar.markdown(f"**Số lượng tài liệu:** {collection_stats.get('count', 0)}")

    # Tùy chọn
    st.sidebar.subheader("Tùy chọn")

    if "evaluate_answers" not in st.session_state:
        st.session_state.evaluate_answers = False

    st.session_state.evaluate_answers = st.sidebar.checkbox(
        "Đánh giá câu trả lời",
        value=st.session_state.evaluate_answers,
        help="Đánh giá chất lượng câu trả lời dựa trên độ liên quan, tính đầy đủ và nhất quán"
    )

    if "show_context" not in st.session_state:
        st.session_state.show_context = False

    st.session_state.show_context = st.sidebar.checkbox(
        "Hiển thị ngữ cảnh",
        value=st.session_state.show_context,
        help="Hiển thị tài liệu tham khảo được sử dụng để trả lời"
    )

    if "show_analysis" not in st.session_state:
        st.session_state.show_analysis = False

    st.session_state.show_analysis = st.sidebar.checkbox(
        "Hiển thị phân tích câu hỏi",
        value=st.session_state.show_analysis,
        help="Hiển thị phân tích về câu hỏi của người dùng"
    )

    # Phần thông tin thêm
    st.sidebar.subheader("Thông tin thêm")
    st.sidebar.markdown("""
    **Công nghệ sử dụng:**
    - LangChain cho xử lý RAG
    - FAISS làm vector database
    - Groq API với LLaMA-70B-8192
    - Hybrid search (cosine + BM25)
    """)

def render_evaluation(evaluation: Dict[str, Any]) -> None:
    """
    Hiển thị kết quả đánh giá câu trả lời.

    Args:
        evaluation (Dict[str, Any]): Kết quả đánh giá
    """
    if not evaluation:
        return

    st.markdown('<div class="eval-metrics">', unsafe_allow_html=True)
    st.markdown("#### 📊 Đánh giá câu trả lời")

    # Các metrics chính
    metrics = []

    if "relevance" in evaluation:
        metrics.append(("Độ liên quan", evaluation["relevance"]["score"]))

    if "completeness" in evaluation:
        metrics.append(("Tính đầy đủ", evaluation["completeness"]["score"]))

    if "consistency" in evaluation:
        metrics.append(("Tính nhất quán", evaluation["consistency"]["score"]))

    if "overall" in evaluation:
        metrics.append(("Tổng thể", evaluation["overall"]))

    if "semantic_similarity" in evaluation:
        metrics.append(("Độ tương đồng ngữ nghĩa", evaluation["semantic_similarity"]))

    # Vẽ biểu đồ
    if metrics:
        fig, ax = plt.subplots(figsize=(8, 4))

        labels = [m[0] for m in metrics]
        values = [m[1] for m in metrics]

        ax.barh(labels, values, color="#2979FF")
        ax.set_xlim(0, 10)
        ax.set_xlabel("Điểm (1-10)")
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        for i, v in enumerate(values):
            ax.text(v + 0.1, i, f"{v:.2f}", va="center")

        plt.tight_layout()
        st.pyplot(fig)

    # Hiển thị giải thích
    if "summary" in evaluation:
        st.markdown(f"**Tóm tắt đánh giá:** {evaluation['summary']}")

    explanation_tabs = st.tabs(["Độ liên quan", "Tính đầy đủ", "Tính nhất quán"])

    with explanation_tabs[0]:
        if "relevance" in evaluation and "explanation" in evaluation["relevance"]:
            st.markdown(evaluation["relevance"]["explanation"])

    with explanation_tabs[1]:
        if "completeness" in evaluation and "explanation" in evaluation["completeness"]:
            st.markdown(evaluation["completeness"]["explanation"])

    with explanation_tabs[2]:
        if "consistency" in evaluation and "explanation" in evaluation["consistency"]:
            st.markdown(evaluation["consistency"]["explanation"])

    st.markdown('</div>', unsafe_allow_html=True)

def render_context(context: List[Dict[str, Any]]) -> None:
    """
    Hiển thị ngữ cảnh tham khảo.

    Args:
        context (List[Dict[str, Any]]): Danh sách tài liệu tham khảo
    """
    if not context:
        return

    st.markdown('<div class="context-box">', unsafe_allow_html=True)
    st.markdown("#### 📚 Tài liệu tham khảo")

    for i, doc in enumerate(context, 1):
        source = doc.get("metadata", {}).get("source", "Không rõ nguồn")

        with st.expander(f"Tài liệu {i} (Nguồn: {source})"):
            st.markdown(doc["text"])

            # Hiển thị score nếu có
            if "score" in doc:
                st.markdown(f"*Độ tương đồng: {doc['score']:.4f}*")

            if "vector_score" in doc and "bm25_score" in doc:
                st.markdown(f"*Vector score: {doc['vector_score']:.4f}, BM25 score: {doc['bm25_score']:.4f}*")

    st.markdown('</div>', unsafe_allow_html=True)

def render_analysis(query: str) -> None:
    """
    Hiển thị phân tích câu hỏi.

    Args:
        query (str): Câu hỏi cần phân tích
    """
    analysis = analyze_query(query, detailed=True)

    if "error" in analysis:
        st.error(f"Lỗi khi phân tích câu hỏi: {analysis['error']}")
        return

    st.markdown('<div class="context-box">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Phân tích câu hỏi")

    # Hiển thị thông tin cơ bản
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Độ dài", analysis.get("length", 0))

    with col2:
        st.metric("Số từ", analysis.get("word_count", 0))

    with col3:
        admission_related = analysis.get("is_admission_related", False)
        st.metric("Liên quan tuyển sinh", "Có" if admission_related else "Không")

    # Hiển thị thông tin chi tiết
    if "main_topic" in analysis:
        st.markdown(f"**Chủ đề chính:** {analysis['main_topic']}")

    if "keywords" in analysis and analysis["keywords"]:
        st.markdown("**Từ khóa:**")
        st.write(", ".join(analysis["keywords"]))

    if "needs_specific_info" in analysis:
        st.markdown(f"**Cần thông tin cụ thể:** {'Có' if analysis['needs_specific_info'] else 'Không'}")

    if "rephrased_question" in analysis:
        st.markdown(f"**Câu hỏi được diễn đạt lại:** {analysis['rephrased_question']}")

    st.markdown('</div>', unsafe_allow_html=True)

def initialize_chat_history() -> None:
    """
    Khởi tạo lịch sử chat nếu chưa có.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_history() -> None:
    """
    Hiển thị lịch sử chat.
    """
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        content = message["content"]
        time_str = message.get("time", "")

        if role == "user":
            st.markdown(f'<div class="chat-message user-message">', unsafe_allow_html=True)
            st.markdown(f'''
                <div class="message-bubble user-bubble">
                    <div class="message-header">
                        <span>👤 Bạn</span>
                        <span class="message-time">{time_str}</span>
                    </div>
                    <div class="message-content">{content}</div>
                </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        else:  # assistant
            st.markdown(f'<div class="chat-message bot-message">', unsafe_allow_html=True)
            st.markdown(f'''
                <div class="message-bubble bot-bubble">
                    <div class="message-header">
                        <span>🎓 Trợ lý</span>
                        <span class="message-time">{time_str}</span>
                    </div>
                    <div class="message-content">{content}</div>
            ''', unsafe_allow_html=True)

            # Hiển thị thông tin nguồn nếu có
            if "source" in message and message["source"] != "LLM":
                st.markdown(f'<div class="source-info">Nguồn: {message["source"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Hiển thị đánh giá và ngữ cảnh
            if st.session_state.evaluate_answers and "evaluation" in message and message["evaluation"]:
                render_evaluation(message["evaluation"])

            if st.session_state.show_context and "context" in message and message["context"]:
                render_context(message["context"])

            if st.session_state.show_analysis:
                render_analysis(st.session_state.messages[i-1]["content"])

    st.markdown('</div>', unsafe_allow_html=True)

def handle_user_input() -> None:
    """
    Xử lý input từ người dùng.
    """
    query = st.chat_input("Nhập câu hỏi của bạn về tuyển sinh...")

    if query:
        # Thêm tin nhắn của người dùng vào lịch sử
        current_time = datetime.now().strftime("%H:%M:%S")

        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "time": current_time
        })

        with st.spinner("Đang xử lý câu hỏi của bạn..."):
            # Gửi câu hỏi đến backend
            response = send_chat_message(query, evaluate=st.session_state.evaluate_answers)

            if "error" in response:
                st.error(f"Lỗi: {response['error']}")
                return

            # Thêm câu trả lời vào lịch sử
            current_time = datetime.now().strftime("%H:%M:%S")

            message = {
                "role": "assistant",
                "content": response.get("answer", "Xin lỗi, không có câu trả lời."),
                "time": current_time,
                "source": response.get("source", "Unknown"),
                "context": response.get("context", []),
                "evaluation": response.get("evaluation", None)
            }

            st.session_state.messages.append(message)

        # Hiển thị lịch sử chat
        display_chat_history()

def main() -> None:
    """
    Hàm main để chạy ứng dụng Streamlit.
    """
    # Khởi tạo
    initialize_chat_history()

    # Render sidebar
    render_sidebar()

    # Tiêu đề trang
    st.markdown('<h1 class="main-header">🎓 Chatbot Tư Vấn Tuyển Sinh</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Hỏi đáp thông tin tuyển sinh với công nghệ AI</p>', unsafe_allow_html=True)

    # Hiển thị lịch sử chat
    display_chat_history()

    # Xử lý input
    handle_user_input()

if __name__ == "__main__":
    main()
