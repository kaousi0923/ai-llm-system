import os
import json
import logging

from modules.config import DEFAULT


def load_history(user_id: str) -> list:
    """
    載入指定使用者的對話歷史，若檔案不存在則回傳空列表。
    """
    history_dir = DEFAULT.get("CHAT_HISTORY_DIR")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f"{user_id}.txt")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error("讀取對話歷史失敗: %s", str(e))
    return []


def save_history(user_id: str, history: list) -> None:
    """
    儲存對話歷史至指定使用者的檔案中。
    """
    history_dir = DEFAULT.get("CHAT_HISTORY_DIR")
    os.makedirs(history_dir, exist_ok=True)
    history_file = os.path.join(history_dir, f"{user_id}.txt")
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception as e:
        logging.error("儲存對話歷史失敗: %s", str(e))
