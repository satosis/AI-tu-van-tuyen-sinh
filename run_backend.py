"""
Script để chạy backend Flask.
"""
import os
import logging
from pathlib import Path

# Thêm đường dẫn gốc vào sys.path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

# Đảm bảo luồng stdout/stderr sử dụng UTF-8 để tránh lỗi UnicodeEncodeError trên Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Cấu hình logging trước khi import các module có thể sử dụng logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend.log', encoding='utf-8')
    ]
)

from app.config import BACKEND_HOST, BACKEND_PORT
from backend.app import app

if __name__ == "__main__":
    print(f"Đang khởi động backend tại http://{BACKEND_HOST}:{BACKEND_PORT}")
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=True)
