import configparser
import os
from dotenv import load_dotenv

# 取得 modules 目錄的絕對路徑，並計算專案根目錄（main.py 與 config.ini 所在目錄）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.ini")
ENV_FILE = os.path.join(PROJECT_DIR, ".env")

# 載入 .env 文件（如果存在）
if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE)
    print(f"✅ 已載入環境配置文件: {ENV_FILE}")
else:
    print(f"⚠️  .env 文件不存在，使用系統環境變數和配置文件")

_config = configparser.ConfigParser()
_config.read(CONFIG_FILE, encoding="utf-8")

# 創建增強的配置字典，支援環境變數優先級
class EnhancedConfigDict:
    def __init__(self, config_section, section_name=""):
        self.config_section = config_section
        self.section_name = section_name
    
    def get(self, key, fallback=None):
        # 優先級: 環境變數 > config.ini > fallback
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        try:
            return self.config_section.get(key, fallback)
        except:
            return fallback

DEFAULT = EnhancedConfigDict(_config["DEFAULT"], "DEFAULT")
EMBED_API_DEFAULTS = EnhancedConfigDict(_config["EMBED_API_DEFAULTS"], "EMBED_API_DEFAULTS")  
CHAT_API_DEFAULTS = EnhancedConfigDict(_config["CHAT_API_DEFAULTS"], "CHAT_API_DEFAULTS")

