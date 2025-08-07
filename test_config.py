#!/usr/bin/env python3
"""
配置測試腳本 - 檢查 .env 文件和 API 密鑰是否正確載入
"""
import os
import sys

def test_configuration():
    print('=' * 50)
    print('[CONFIG TEST] RAG 系統配置測試')
    print('=' * 50)
    print()
    
    # 檢查 .env 文件是否存在
    env_file = '.env'
    if os.path.exists(env_file):
        print(f'[OK] .env 文件存在: {env_file}')
    else:
        print(f'[ERROR] .env 文件不存在: {env_file}')
        print('[TIP] 運行 setup_local_dev.bat 創建配置文件')
        return False
    
    print()
    
    # 測試配置載入
    try:
        from modules.config import DEFAULT
        print('[OK] 配置模組載入成功')
        
        # 檢查重要的配置項
        configs_to_check = {
            'OPENAI_API_KEY': 'OpenAI API 密鑰',
            'HF_TOKEN': 'HuggingFace Token', 
            'VECTOR_DBS_DIR': '向量資料庫目錄',
            'TRAINED_MODELS_DIR': '訓練模型目錄'
        }
        
        print()
        print('[CONFIG] 配置項檢查結果:')
        print('-' * 30)
        
        all_good = True
        for key, description in configs_to_check.items():
            value = DEFAULT.get(key)
            if value:
                # 對於敏感資訊只顯示前幾個字符
                if 'KEY' in key or 'TOKEN' in key:
                    if len(value) > 10:
                        display_value = f"{value[:4]}...{value[-4:]}"
                    else:
                        display_value = f"{value[:3]}..."
                else:
                    display_value = value
                print(f'[OK] {description}: {display_value}')
            else:
                print(f'[ERROR] {description}: 未設置')
                all_good = False
        
        print()
        if all_good:
            print('[SUCCESS] 所有配置項檢查通過！系統可以正常運行')
            print('[TIP] 下一步: 執行 llm_api_start.bat 啟動系統')
        else:
            print('[WARNING] 部分配置項缺失，可能會影響系統功能')
            print('[TIP] 建議: 檢查並完善 .env 文件中的配置')
        
    except ImportError as e:
        print(f'[ERROR] 配置模組載入失敗: {e}')
        print('[TIP] 提示: 確保已安裝所有依賴 (python-dotenv)')
        return False
    except Exception as e:
        print(f'[ERROR] 配置測試失敗: {e}')
        return False
    
    print()
    print('=' * 50)
    return True

if __name__ == "__main__":
    success = test_configuration()
    input("按 Enter 鍵退出...")
    sys.exit(0 if success else 1)