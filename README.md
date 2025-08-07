# AI/LLM RAG 系統

<div align="center">
  <img src="img/MIRDC_Logo.png" alt="MIRDC Logo" width="200"/>
  
  [![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.112.0-green.svg)](https://fastapi.tiangolo.com/)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
</div>

## 📝 專案簡介

這是一個基於 FastAPI 的先進 RAG (Retrieval-Augmented Generation) 系統，整合了向量資料庫、多模型支援、以及 LoRA 微調功能。系統支援本地模型（Gemma）和雲端模型（OpenAI GPT），提供完整的文件處理、語義搜索、和智能對話功能。

### ✨ 主要特色

- 🤖 **多模型支援**：整合 OpenAI GPT-4 系列和本地 Gemma 模型
- 📚 **混合搜索**：結合 BM25 關鍵字搜索和語義向量搜索
- 🎯 **LoRA 微調**：支援本地模型的快速微調訓練
- 🖼️ **多模態支援**：支援圖片輸入（限 OpenAI 模型）
- 📂 **多格式文件處理**：支援 PDF、DOCX、TXT、CSV、XLSX
- 🔍 **智能檢索**：自動實體提取和查詢重寫優化
- 💾 **對話歷史管理**：支援多用戶對話記錄
- 🌐 **外網穿透**：內建 Cloudflare 和 Bore 隧道支援

## 🚀 快速開始

### 系統需求

- Windows 10/11
- Python 3.10
- CUDA 11.8+ (用於本地模型)
- Anaconda 或 Miniconda
- 至少 8GB RAM (建議 16GB+)
- GPU with 6GB+ VRAM (用於本地模型)

### 安裝步驟

1. **克隆專案**
```bash
git clone https://github.com/kaousi0923/ai-llm-rag.git
cd ai-llm-rag
```

2. **設置環境變數**
```bash
# 複製環境變數範例檔案
copy .env.example .env
# 編輯 .env 檔案，填入您的 API 金鑰
notepad .env
```

3. **安裝依賴套件**

使用提供的批次檔：
```bash
# 安裝並啟動系統
llm_install.bat
```

或手動安裝：
```bash
# 創建 conda 環境
conda create -n MIRDC_Unsloth_clone python=3.10 -y
conda activate MIRDC_Unsloth_clone

# 安裝依賴套件
pip install -r requirements1.txt
pip install -r requirements2.txt
pip install -r requirements3.txt
```

4. **啟動服務**
```bash
# 使用批次檔
llm_api_start.bat

# 或直接執行
python main.py
```

服務將在 `http://localhost:5000` 啟動

## 📖 使用指南

### API 文檔

啟動服務後，訪問以下網址查看互動式 API 文檔：
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

### 核心功能

#### 1. 建立向量資料庫

```python
import requests

# 上傳文件並建立向量資料庫
files = {'files': open('document.pdf', 'rb')}
data = {
    'name': '知識庫名稱',
    'model': 'intfloat/multilingual-e5-base'
}
response = requests.post('http://localhost:5000/api/embed', 
                         files=files, data=data)
```

#### 2. 智能對話

```python
# 使用 RAG 進行對話
data = {
    'model': 'gpt-4o-mini',
    'messages': '請問系統支援哪些功能？',
    'database': '知識庫名稱',
    'temperature': '0.5',
    'user_id': 'user123'  # 可選，用於保存對話歷史
}
response = requests.post('http://localhost:5000/api/chat', data=data)
```

#### 3. 模型微調

```python
# 使用 LoRA 微調本地模型
files = {'file': open('training_data.xlsx', 'rb')}
data = {
    'model': 'unsloth/gemma-3-1b-it-unsloth-bnb-4bit',
    'new_model': '我的微調模型',
    'epochs': 5,
    'learning_rate': 1e-5
}
response = requests.post('http://localhost:5000/api/train', 
                         files=files, data=data)
```

### 支援的模型

**雲端模型 (需要 API Key)**
- GPT-4o-mini
- GPT-4.1
- ChatGPT-4o-latest

**本地模型 (需要 GPU)**
- unsloth/gemma-3-1b-it
- unsloth/gemma-3-4b-it
- 自訂微調模型

## 🏗️ 專案結構

```
ai_llm/
├── main.py                 # 主程式入口
├── config.ini             # 配置檔案
├── modules/               # 核心模組
│   ├── config.py         # 配置管理
│   ├── file_processing.py # 文件處理
│   ├── vector_db.py      # 向量資料庫
│   ├── chat_history.py   # 對話歷史
│   └── llm_caller_unsloth.py # 模型調用
├── utils/                 # 工具函式
├── tests/                 # 測試套件
├── vector_dbs/           # 向量資料庫存儲
├── trained_models/       # 微調模型存儲
├── logs/                 # 系統日誌
└── dependencies/         # 外部依賴
```

## 🧪 測試

執行測試套件：
```bash
# 安裝測試依賴
pip install -r requirements-test.txt

# 執行所有測試
pytest

# 執行特定測試
pytest tests/test_api.py -v

# 生成覆蓋率報告
pytest --cov=. --cov-report=html
```

## 🔧 配置說明

### 環境變數

關鍵環境變數配置（參見 `.env.example`）：

```env
# API 金鑰
OPENAI_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token

# 模型配置
MAX_SEQ_LENGTH=2048
BASE_MODELS=model1,model2

# 目錄設定
VECTOR_DBS_DIR=vector_dbs
TRAINED_MODELS_DIR=trained_models
```

### 進階配置

編輯 `config.ini` 進行進階設定：

```ini
[DEFAULT]
max_seq_length = 2048
default_prompt_template = 你的提示模板

[EMBED_API_DEFAULTS]
chunk_size = 1500
chunk_overlap = 100

[CHAT_API_DEFAULTS]
temperature = 0.1
max_tokens = 2048
```

## 🚧 開發指南

### 新增自訂模型

1. 在 `config.ini` 中新增模型名稱
2. 實作模型載入邏輯（如需要）
3. 更新 API 文檔

### 擴充文件處理格式

1. 在 `modules/file_processing.py` 新增處理函式
2. 更新支援格式列表
3. 新增對應測試案例

### API 擴充

1. 在 `main.py` 新增 endpoint
2. 更新 API 文檔
3. 撰寫對應測試

## 📊 效能優化

- **GPU 記憶體管理**：系統自動管理模型載入/卸載
- **批次處理**：支援多文件同時處理
- **快取機制**：使用 Hugging Face 快取加速模型載入
- **非同步處理**：FastAPI 非同步架構提升並發性能

## 🔒 安全性考量

- ⚠️ **請勿將 API 金鑰寫入程式碼**
- 使用環境變數管理敏感資訊
- 定期更新依賴套件
- 實施輸入驗證和錯誤處理
- 考慮新增 API 認證機制

## 🤝 貢獻指南

歡迎貢獻程式碼！請遵循以下步驟：

1. Fork 專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📝 待辦事項

- [ ] 新增 API 認證/授權機制
- [ ] 實施結構化日誌系統
- [ ] 新增 Docker 容器化支援
- [ ] 完善單元測試覆蓋率
- [ ] 新增效能監控儀表板
- [ ] 支援更多檔案格式
- [ ] 實施資料庫備份策略
- [ ] 新增 CI/CD 流程

## 🐛 問題回報

如遇到問題，請在 [Issues](https://github.com/kaousi0923/issues) 頁面回報，並提供：
- 問題描述
- 重現步驟
- 錯誤訊息
- 系統環境資訊

## 👥 團隊

- 開發團隊：MIRDC
- 維護者：[kaousi0923]

## 🙏 致謝

- [FastAPI](https://fastapi.tiangolo.com/)
- [Unsloth](https://github.com/unslothai/unsloth)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/)

---

<div align="center">
  Made with ❤️ by MIRDC Team
</div>
