"""
Pytest configuration and fixtures
"""
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI app"""
    from main import app
    return TestClient(app)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing"""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("This is a sample text file for testing purposes.")
    return file_path


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration for testing"""
    return {
        "DEFAULT": {
            "LOGS_DIR": "test_logs",
            "VECTOR_DBS_DIR": "test_vector_dbs",
            "CHAT_HISTORY_DIR": "test_chat_history",
            "TEMP_DIR": "test_temp",
            "TRAINED_MODELS_DIR": "test_models",
            "MAX_SEQ_LENGTH": "2048"
        },
        "EMBED_API_DEFAULTS": {
            "MODEL": "intfloat/multilingual-e5-base",
            "TEXT_SPLITTER": "Recursive",
            "CHUNK_SIZE": "1500",
            "CHUNK_OVERLAP": "100",
            "SUMMARY_THRESHOLD": "100000"
        },
        "CHAT_API_DEFAULTS": {
            "MODEL": "gpt-4o-mini",
            "TEMPERATURE": "0.1",
            "TOP_P": "0.1",
            "MAX_TOKENS": "2048",
            "HISTORY_LENGTH": "3",
            "RETRIEVAL_K": "0.5"
        }
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {
        "choices": [{
            "message": {
                "content": "This is a mock response from OpenAI"
            }
        }]
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Setup test environment variables"""
    monkeypatch.setenv("PYTHONUTF8", "1")
    monkeypatch.setenv("DISABLE_TORCH_INDUCTOR", "1")
    monkeypatch.setenv("TORCHINDUCTOR_DISABLE", "1")
    monkeypatch.setenv("TORCH_COMPILE", "0")
    monkeypatch.setenv("HF_HUB_ENABLE_HF_TRANSFER", "0")
    monkeypatch.setenv("TUNNEL_PORT", "5001")  # Use different port for testing


@pytest.fixture
def sample_qa_data():
    """Sample Q&A data for testing"""
    return [
        {"Q": "什麼是機器學習？", "A": "機器學習是人工智慧的一個分支"},
        {"Q": "如何使用API？", "A": "請參考API文檔進行操作"},
        {"Q": "系統支援哪些模型？", "A": "支援GPT-4和本地Gemma模型"}
    ]