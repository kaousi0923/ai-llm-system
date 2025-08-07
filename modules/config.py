import configparser
import os

# 取得 modules 目錄的絕對路徑，並計算專案根目錄（main.py 與 config.ini 所在目錄）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.ini")

_config = configparser.ConfigParser()
_config.read(CONFIG_FILE, encoding="utf-8")

DEFAULT = _config["DEFAULT"]
EMBED_API_DEFAULTS = _config["EMBED_API_DEFAULTS"]
CHAT_API_DEFAULTS = _config["CHAT_API_DEFAULTS"]

