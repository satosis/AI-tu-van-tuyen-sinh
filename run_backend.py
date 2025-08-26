"""
Script để chạy backend Flask.
"""
import os
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
        logging.FileHandler('backend.log')
    ]
)

from app.config import BACKEND_HOST, BACKEND_PORT
from backend.app import app

if __name__ == "__main__":
    print(f"Đang khởi động backend tại http://{BACKEND_HOST}:{BACKEND_PORT}")
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=True)
