"""
Script để chạy frontend Streamlit.
"""
import os
import subprocess
import logging
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

# Cấu hình logging trước khi import các module có thể sử dụng logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('frontend.log')
    ]
)

from app.config import FRONTEND_PORT

if __name__ == "__main__":
    print(f"Đang khởi động frontend tại http://localhost:{FRONTEND_PORT}")

    # Chạy ứng dụng Streamlit
    frontend_path = Path(__file__).resolve().parent / "frontend" / "app.py"

    subprocess.run([
        "streamlit", "run",
        str(frontend_path),
        "--server.port", str(FRONTEND_PORT),
        "--browser.serverAddress", "localhost",
        "--server.headless", "true"
    ])

