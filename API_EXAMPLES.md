# API 使用範例

這個文件提供了完整的 API 使用範例，幫助你快速上手。

## 🚀 基本設定

### Python 客戶端設定

```python
import requests
import json
from pathlib import Path

# API 基礎 URL
BASE_URL = "http://localhost:5000"

# 設定標頭
headers = {
    'Content-Type': 'application/json'
}
```

### cURL 設定

```bash
# 基礎 URL
export BASE_URL="http://localhost:5000"
```

## 📚 1. 向量資料庫管理

### 建立向量資料庫

**Python 範例**：
```python
def create_vector_database():
    """建立新的向量資料庫"""
    url = f"{BASE_URL}/api/embed"
    
    # 準備檔案
    files = {
        'files': [
            ('document1.pdf', open('path/to/document1.pdf', 'rb')),
            ('document2.docx', open('path/to/document2.docx', 'rb'))
        ]
    }
    
    # 設定參數
    data = {
        'name': '企業知識庫',
        'model': 'intfloat/multilingual-e5-base',
        'summarizer': '這是企業內部文件知識庫',
        'chunk_size': '1500',
        'chunk_overlap': '100'
    }
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 資料庫建立成功")
        print(f"摘要: {result['summary']}")
    else:
        print(f"❌ 建立失敗: {response.text}")

# 執行
create_vector_database()
```

**cURL 範例**：
```bash
curl -X POST "$BASE_URL/api/embed" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "name=企業知識庫" \
  -F "model=intfloat/multilingual-e5-base" \
  -F "chunk_size=1500"
```

### 列出所有向量資料庫

**Python 範例**：
```python
def list_databases():
    """列出所有向量資料庫"""
    url = f"{BASE_URL}/api/embed/list"
    
    response = requests.get(url)
    databases = response.json()
    
    print("📚 可用的向量資料庫:")
    for db in databases:
        print(f"  • {db['name']} (建立於: {db['created_at']})")
        print(f"    摘要: {db['summary']}")
        print(f"    檔案: {', '.join(db['files'])}")
        print()

list_databases()
```

### 刪除向量資料庫

**Python 範例**：
```python
def delete_database(db_name):
    """刪除指定的向量資料庫"""
    url = f"{BASE_URL}/api/embed/delete"
    params = {'name': db_name}
    
    response = requests.delete(url, params=params)
    result = response.json()
    
    if result['status'] == 'success':
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['message']}")

# 刪除資料庫
delete_database('舊的知識庫')
```

## 💬 2. 對話功能

### 基本對話

**Python 範例**：
```python
def simple_chat(message, model="gpt-4o-mini"):
    """簡單對話（不使用 RAG）"""
    url = f"{BASE_URL}/api/chat"
    
    data = {
        'model': model,
        'messages': message,
        'temperature': '0.7'
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        result = response.json()
        return result['content']
    else:
        return f"錯誤: {response.text}"

# 使用範例
answer = simple_chat("什麼是機器學習？")
print(f"🤖 回答: {answer}")
```

### RAG 對話

**Python 範例**：
```python
def rag_chat(question, database_name, user_id=None):
    """使用 RAG 的對話"""
    url = f"{BASE_URL}/api/chat"
    
    data = {
        'model': 'gpt-4o-mini',
        'messages': question,
        'database': database_name,
        'temperature': '0.3',
        'retrieval_k': '0.7',
        'user_id': user_id  # 可選：保存對話歷史
    }
    
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        result = response.json()
        return result['content']
    else:
        return f"錯誤: {response.text}"

# 使用範例
question = "公司的環保政策是什麼？"
answer = rag_chat(question, "企業知識庫", user_id="user001")
print(f"📋 問題: {question}")
print(f"🤖 回答: {answer}")
```

### 多模態對話（圖片 + 文字）

**Python 範例**：
```python
def multimodal_chat(message, image_path):
    """多模態對話（文字 + 圖片）"""
    url = f"{BASE_URL}/api/chat"
    
    # 準備檔案
    files = {
        'image': open(image_path, 'rb')
    }
    
    # 設定參數
    data = {
        'model': 'gpt-4o-mini',  # 必須使用支援視覺的模型
        'messages': message,
        'enable_vision': True,
        'temperature': '0.5'
    }
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        return result['content']
    else:
        return f"錯誤: {response.text}"

# 使用範例
answer = multimodal_chat(
    "請描述這張圖片的內容", 
    "screenshot.png"
)
print(f"🖼️ 圖片分析: {answer}")
```

## 🎯 3. 模型微調

### LoRA 微調

**Python 範例**：
```python
def train_model():
    """微調本地模型"""
    url = f"{BASE_URL}/api/train"
    
    # 準備訓練資料檔案 (Excel 格式，包含 Q 和 A 欄位)
    files = {
        'file': open('training_data.xlsx', 'rb')
    }
    
    # 訓練參數
    data = {
        'model': 'unsloth/gemma-3-1b-it-unsloth-bnb-4bit',
        'new_model': '客服機器人v1',
        'epochs': 3,
        'learning_rate': 1e-5,
        'batch_size': 16,
        'r': 64,  # LoRA rank
        'dropout': 0.05
    }
    
    print("🚀 開始模型微調...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 微調完成! 模型名稱: {result['model_name']}")
    else:
        print(f"❌ 微調失敗: {response.text}")

# 準備訓練資料格式範例
import pandas as pd

training_data = pd.DataFrame({
    'Q': [
        '你好',
        '公司地址在哪裡？',
        '如何申請退貨？'
    ],
    'A': [
        '您好！我是客服助理，很高興為您服務。',
        '公司地址：台北市信義區信義路五段7號',
        '您可以在購買後30天內申請退貨，請聯繫客服專線。'
    ]
})

training_data.to_excel('training_data.xlsx', index=False)
print("📝 訓練資料已準備完成")

# 執行微調
train_model()
```

### 列出可用模型

**Python 範例**：
```python
def list_models():
    """列出所有可用模型"""
    url = f"{BASE_URL}/api/model/list"
    
    response = requests.get(url)
    models = response.json()
    
    print("🤖 可用的模型:")
    
    # 分類顯示
    base_models = [m for m in models if m.get('lora_name') is None]
    fine_tuned_models = [m for m in models if m.get('lora_name') is not None]
    
    print("\n📦 基礎模型:")
    for model in base_models:
        print(f"  • {model['name']}")
    
    print("\n🎯 微調模型:")
    for model in fine_tuned_models:
        print(f"  • {model['name']} (基於 {model['base_model']})")
        print(f"    建立時間: {model['created_at']}")

list_models()
```

## 🔄 4. 資料擴增

### 問答資料擴增

**Python 範例**：
```python
def augment_data():
    """擴增問答資料"""
    url = f"{BASE_URL}/api/data/augment"
    
    # 準備原始資料
    files = {
        'file': open('original_qa.xlsx', 'rb')
    }
    
    data = {
        'api_key': 'your-openai-api-key',
        'model_name': 'gpt-4o-mini',
        'expand_count': 5,  # 每個 Q&A 擴增為 5 個變體
        'temperature': 0.7
    }
    
    print("🔄 開始資料擴增...")
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        # 下載擴增後的檔案
        with open('augmented_data.xlsx', 'wb') as f:
            f.write(response.content)
        print("✅ 資料擴增完成！檔案已儲存為 augmented_data.xlsx")
    else:
        print(f"❌ 擴增失敗: {response.text}")

augment_data()
```

## 🔧 5. 進階用法

### 批次處理對話

**Python 範例**：
```python
def batch_chat(questions, database_name):
    """批次處理多個問題"""
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"處理問題 {i}/{len(questions)}: {question[:50]}...")
        
        answer = rag_chat(question, database_name)
        results.append({
            'question': question,
            'answer': answer
        })
    
    return results

# 使用範例
questions = [
    "公司的核心價值是什麼？",
    "如何申請年假？",
    "員工福利有哪些？"
]

results = batch_chat(questions, "HR知識庫")

for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print("-" * 50)
```

### 串流對話

**Python 範例**：
```python
def streaming_chat(user_id):
    """模擬串流對話"""
    conversation_history = []
    
    print("💬 開始對話（輸入 'quit' 結束）")
    
    while True:
        user_input = input("您: ")
        if user_input.lower() == 'quit':
            break
        
        # 發送訊息
        answer = rag_chat(user_input, "企業知識庫", user_id)
        print(f"🤖: {answer}")
        
        # 記錄對話
        conversation_history.append({
            'user': user_input,
            'assistant': answer
        })
    
    print("\n📝 對話記錄已自動儲存")
    return conversation_history

# 使用範例
history = streaming_chat("user123")
```

### 錯誤處理範例

**Python 範例**：
```python
def robust_api_call(func, *args, max_retries=3, **kwargs):
    """帶重試機制的 API 呼叫"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            print(f"⚠️ 第 {attempt + 1} 次嘗試失敗: {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # 指數退避

# 使用範例
try:
    result = robust_api_call(rag_chat, "測試問題", "知識庫")
    print(f"✅ 成功: {result}")
except Exception as e:
    print(f"❌ 最終失敗: {e}")
```

## 📊 6. 監控和統計

### 系統狀態檢查

**Python 範例**：
```python
def check_system_status():
    """檢查系統狀態"""
    checks = []
    
    # 檢查 API 可用性
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        checks.append(("API 服務", response.status_code == 200))
    except:
        checks.append(("API 服務", False))
    
    # 檢查模型列表
    try:
        models = requests.get(f"{BASE_URL}/api/model/list").json()
        checks.append(("模型服務", len(models) > 0))
    except:
        checks.append(("模型服務", False))
    
    # 檢查資料庫列表
    try:
        dbs = requests.get(f"{BASE_URL}/api/embed/list").json()
        checks.append(("資料庫服務", isinstance(dbs, list)))
    except:
        checks.append(("資料庫服務", False))
    
    # 顯示結果
    print("🔍 系統狀態檢查:")
    for service, status in checks:
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {service}: {'正常' if status else '異常'}")

check_system_status()
```

## 🚨 常見問題解決

### 1. 連線問題

```python
# 檢查服務是否啟動
def check_connection():
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        print("✅ 服務正常運行")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ 無法連線到服務，請確認：")
        print("  1. 服務是否已啟動 (python main.py)")
        print("  2. 埠號是否正確 (預設 5000)")
        return False

check_connection()
```

### 2. API 金鑰問題

```python
# 測試 OpenAI API 金鑰
def test_openai_key():
    result = simple_chat("測試訊息", "gpt-4o-mini")
    if "錯誤" in result:
        print("❌ OpenAI API 金鑰可能有問題")
        print("請檢查 .env 檔案中的 OPENAI_API_KEY")
    else:
        print("✅ OpenAI API 金鑰正常")

test_openai_key()
```

---

更多範例和詳細說明，請參考 [API 文檔](http://localhost:5000/docs)