# 測試指南

## 🧪 快速開始測試

### 方法一：使用批次檔（推薦）

```bash
# 執行測試批次檔
run_tests.bat
```

然後選擇測試模式：
- 1：執行所有測試
- 2：執行單元測試（快速）
- 3：執行 API 測試
- 4：執行整合測試
- 5：生成覆蓋率報告
- 6：執行特定測試檔案

### 方法二：手動執行測試

```bash
# 啟用虛擬環境
conda activate MIRDC_Unsloth_clone

# 安裝測試套件
pip install -r requirements-test.txt

# 執行所有測試
pytest

# 執行特定測試檔案
pytest tests/test_api.py -v

# 只執行標記為 unit 的測試
pytest -m unit

# 生成覆蓋率報告
pytest --cov=. --cov-report=html
```

## 📁 測試結構

```
tests/
├── conftest.py          # 測試配置和共用 fixtures
├── test_api.py          # API 端點測試
├── test_config.py       # 配置管理測試
├── test_file_processing.py  # 檔案處理測試
└── test_vector_db.py    # 向量資料庫測試
```

## 🏷️ 測試標記

測試使用以下標記分類：

- `@pytest.mark.unit` - 單元測試（快速、獨立）
- `@pytest.mark.integration` - 整合測試（較慢、需要外部資源）
- `@pytest.mark.api` - API 端點測試
- `@pytest.mark.slow` - 慢速測試
- `@pytest.mark.skip` - 跳過的測試

## 🔧 測試環境準備

### 1. 建立測試用 .env 檔案

```bash
# 複製範例檔案
copy .env.example .env.test

# 編輯測試環境變數
notepad .env.test
```

### 2. 準備測試資料

在 `tests/test_data/` 建立測試檔案：
```
tests/test_data/
├── sample.pdf
├── sample.docx
├── sample.txt
└── qa_data.xlsx
```

### 3. Mock 外部服務

測試會自動 mock OpenAI API，不需要真實 API key。

## 📊 測試覆蓋率

### 查看覆蓋率報告

```bash
# 生成終端機報告
pytest --cov=. --cov-report=term-missing

# 生成 HTML 報告
pytest --cov=. --cov-report=html
# 開啟 htmlcov/index.html 查看
```

### 目標覆蓋率

- 核心功能：> 80%
- API 端點：> 90%
- 工具函數：> 95%

## 🐛 測試偵錯

### 顯示詳細輸出

```bash
# 顯示所有 print 輸出
pytest -s

# 顯示詳細測試資訊
pytest -vv

# 在第一個失敗時停止
pytest -x

# 顯示本地變數
pytest -l
```

### 執行特定測試

```bash
# 執行單一測試函數
pytest tests/test_api.py::TestAPIEndpoints::test_root_endpoint

# 執行包含關鍵字的測試
pytest -k "vector"

# 執行上次失敗的測試
pytest --lf
```

## ⚠️ 常見問題

### 1. ImportError

**問題**：`ModuleNotFoundError: No module named 'main'`

**解決**：確保在專案根目錄執行測試
```bash
cd C:\Users\scai\Desktop\ai_llm
pytest
```

### 2. 缺少測試依賴

**問題**：`ModuleNotFoundError: No module named 'pytest'`

**解決**：安裝測試套件
```bash
pip install -r requirements-test.txt
```

### 3. GPU/CUDA 錯誤

**問題**：測試時出現 CUDA 錯誤

**解決**：在測試時停用 GPU
```python
# 在 conftest.py 中加入
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### 4. 測試資料庫衝突

**問題**：測試時與實際資料庫衝突

**解決**：使用獨立的測試目錄
```python
# conftest.py
@pytest.fixture(autouse=True)
def setup_test_dirs(monkeypatch):
    monkeypatch.setenv("VECTOR_DBS_DIR", "test_vector_dbs")
    monkeypatch.setenv("LOGS_DIR", "test_logs")
```

## 📝 撰寫新測試

### 測試命名規範

```python
# 檔案名稱：test_<module_name>.py
# 類別名稱：Test<FeatureName>
# 函數名稱：test_<what_it_does>

class TestVectorDatabase:
    def test_create_database_success(self):
        """測試成功建立資料庫"""
        pass
    
    def test_create_database_with_invalid_name(self):
        """測試使用無效名稱建立資料庫"""
        pass
```

### 使用 Fixtures

```python
# conftest.py 中定義共用 fixture
@pytest.fixture
def sample_documents():
    return [
        "Document 1 content",
        "Document 2 content"
    ]

# 在測試中使用
def test_process_documents(sample_documents):
    result = process(sample_documents)
    assert len(result) == 2
```

### Mock 外部服務

```python
from unittest.mock import patch, MagicMock

@patch('main.client.chat.completions.create')
def test_openai_call(mock_create):
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Test response"))]
    )
    
    # 執行測試
    response = call_openai("test prompt")
    assert response == "Test response"
```

## 🚀 持續整合

### GitHub Actions 配置範例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'
    - name: Install dependencies
      run: |
        pip install -r requirements1.txt
        pip install -r requirements-test.txt
    - name: Run tests
      run: pytest --cov=. --cov-report=xml
    - name: Upload coverage
      uses: codecov/codecov-action@v1
```

## 📈 測試指標

定期檢查以下指標：

- **測試數量**：目標 > 100 個測試
- **執行時間**：單元測試 < 10 秒
- **覆蓋率**：整體 > 70%
- **失敗率**：< 1%

## 🔍 進階測試

### 效能測試

```python
import pytest
import time

@pytest.mark.slow
def test_performance():
    start = time.time()
    # 執行操作
    result = expensive_operation()
    duration = time.time() - start
    
    assert duration < 5.0  # 應在 5 秒內完成
```

### 壓力測試

```python
import concurrent.futures

def test_concurrent_requests():
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(api_call) for _ in range(100)]
        results = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in results)
```

## 💡 測試最佳實踐

1. **獨立性**：每個測試應該獨立執行
2. **可重複**：測試結果應該一致
3. **快速**：單元測試應該快速執行
4. **清晰**：測試名稱應該描述測試內容
5. **完整**：測試正常和異常情況

---

更多測試相關問題，請參考 [pytest 文檔](https://docs.pytest.org/)