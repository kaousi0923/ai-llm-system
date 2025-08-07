# 貢獻指南

## 🤝 歡迎貢獻

感謝您對 AI/LLM RAG 系統的貢獻！本指南將幫助您順利參與專案開發。

## 🚀 開發環境設置

### 1. 複製專案
```bash
git clone https://github.com/your-repo/ai-llm-rag.git
cd ai-llm-rag
```

### 2. 環境配置
```bash
# 複製環境變數檔案
copy .env.example .env

# 編輯並填入您的 API 金鑰
notepad .env

# 安裝開發環境
llm_install.bat
```

### 3. 安裝開發工具
```bash
conda activate MIRDC_Unsloth_clone
pip install -r requirements-dev.txt
```

## 📝 開發流程

### 分支策略
- `main` - 穩定版本
- `develop` - 開發版本
- `feature/xxx` - 新功能分支
- `bugfix/xxx` - 錯誤修復分支

### 提交流程
1. 創建功能分支
```bash
git checkout -b feature/your-feature-name
```

2. 進行開發並測試
```bash
# 執行測試
run_tests.bat

# 檢查程式碼風格
black .
flake8 .
```

3. 提交更改
```bash
git add .
git commit -m "feat: 新增 XXX 功能"
```

4. 推送並建立 Pull Request
```bash
git push origin feature/your-feature-name
```

## 📋 程式碼規範

### Python 風格
- 遵循 PEP 8
- 使用 Black 格式化
- 行長度限制：88 字元
- 使用 type hints

### 提交訊息格式
```
type(scope): 簡短描述

詳細描述（可選）

- 相關 issue: #123
- 破壞性變更: 是/否
```

**類型**：
- `feat`: 新功能
- `fix`: 錯誤修復
- `docs`: 文檔更新
- `style`: 程式碼風格
- `refactor`: 重構
- `test`: 測試相關
- `chore`: 其他變更

### 檔案結構
```python
"""
模組說明

這個模組用來...
"""
import os
import sys
from typing import Optional, List

# 常數定義
CONSTANT_VALUE = "value"

class ExampleClass:
    """類別說明"""
    
    def __init__(self, param: str):
        """初始化方法"""
        self.param = param
    
    def method(self, arg: Optional[str] = None) -> bool:
        """方法說明
        
        Args:
            arg: 參數說明
            
        Returns:
            返回值說明
            
        Raises:
            ValueError: 錯誤說明
        """
        pass
```

## 🧪 測試要求

### 測試覆蓋率
- 新功能：100% 覆蓋率
- 核心模組：> 80%
- API 端點：> 90%

### 測試類型
```python
# 單元測試
@pytest.mark.unit
def test_function_success():
    """測試函數正常情況"""
    pass

# API 測試
@pytest.mark.api
def test_api_endpoint():
    """測試 API 端點"""
    pass

# 整合測試
@pytest.mark.integration
def test_integration():
    """測試整合功能"""
    pass
```

## 📖 文檔要求

### API 文檔
- 使用 FastAPI 自動生成
- 完整的參數說明
- 範例請求/回應

### 程式碼文檔
- 所有公共函數都要有 docstring
- 複雜邏輯要有註釋
- README 保持更新

## 🐛 錯誤回報

### Bug Report 範本
```markdown
## Bug 描述
簡潔描述問題

## 重現步驟
1. 執行 ...
2. 輸入 ...
3. 看到錯誤 ...

## 預期行為
應該發生什麼

## 實際行為
實際發生什麼

## 環境資訊
- OS: Windows 11
- Python: 3.10
- 版本: v1.0.0

## 錯誤訊息
```
paste error message here
```

## 額外資訊
其他相關資訊
```

### Feature Request 範本
```markdown
## 功能描述
清楚描述建議的功能

## 使用情境
描述為什麼需要這個功能

## 解決方案
建議的實作方式

## 替代方案
其他可能的解決方案

## 額外資訊
其他相關資訊
```

## 🔍 程式碼審查

### 審查重點
- [ ] 功能正確性
- [ ] 測試覆蓋率
- [ ] 程式碼風格
- [ ] 效能考量
- [ ] 安全性檢查
- [ ] 文檔完整性

### 審查清單
- [ ] 程式碼通過所有測試
- [ ] 沒有明顯的效能問題
- [ ] 遵循專案的程式碼風格
- [ ] 有適當的錯誤處理
- [ ] API 向下相容
- [ ] 文檔已更新

## 🏆 認可貢獻者

所有貢獻者都會在以下地方被認可：
- README.md 的貢獻者清單
- CHANGELOG.md 的版本說明
- GitHub Contributors 頁面

## 📞 聯絡方式

- 技術問題：建立 GitHub Issue
- 一般討論：GitHub Discussions
- 緊急問題：email@example.com

## 📄 授權

貢獻的程式碼將使用與專案相同的 MIT 授權。

---

再次感謝您的貢獻！ 🎉