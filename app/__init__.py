"""
Package initialization for app.
"""
import logging
from logging.config import dictConfig

from app.config import LOG_LEVEL

# Cấu hình logging
logging_config = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': LOG_LEVEL,
            'formatter': 'standard',
            'filename': 'app.log',
            'mode': 'a',
            'encoding': 'utf-8',
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True
        }
    }
}

dictConfig(logging_config)
logger = logging.getLogger(__name__)
logger.info("Khởi tạo ứng dụng")
