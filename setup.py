"""
Setup script cho việc cài đặt package của dự án chatbot tư vấn tuyển sinh.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="admission-chatbot",
    version="0.1.0",
    author="Đồ Án",
    author_email="your_email@example.com",
    description="Chatbot tư vấn tuyển sinh sử dụng RAG và LLM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/admission-chatbot",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit==1.34.0",
        "flask==3.0.0",
        "gunicorn==23.0.0",
        "langchain==0.3.1",
        "langchain-community==0.3.1",
        "langchain-groq==0.3.2",
        "langchain-text-splitters>=0.3.8",
        "faiss-cpu==1.7.4",
        "transformers==4.51.3",
        "sentence-transformers==4.1.0",
        "torch==2.7.0",
        "pypdf==3.17.1",
        "scikit-learn==1.6.1",
        "numpy==1.26.2",
        "pandas==2.1.3",
        "matplotlib==3.8.2",
        "python-dotenv==1.0.0",
        "requests==2.32.3",
        "packaging==23.2",
        "pillow<11.0.0",
        "tornado==6.2",
        "typing-extensions==4.7.1",
        "pydantic==2.7.4",
    ],
    entry_points={
        "console_scripts": [
            "run-backend=run_backend:main",
            "run-frontend=run_frontend:main",
            "ingest-data=scripts.ingest:main",
            "evaluate-bot=scripts.evaluate:main",
            "train-embeddings=scripts.train_embeddings:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.env"],
    },
)