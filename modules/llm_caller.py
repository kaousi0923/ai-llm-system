import json
import logging
import requests

from modules.config import DEFAULT


def call_llm_for_summary(prompt: str, text: str) -> str:
    """
    使用 LLM 進行摘要，要求摘要以繁體中文呈現。
    """
    url = DEFAULT.get("LLM_API_URL")
    payload = {
        "model": "gemma3:27b-it-q4_K_M",
        "messages": [
            {
                "role": "system",
                "content": "你是一個使用繁體中文的文件摘要助理，請根據提示詞完成摘要。"
            },
            {
                "role": "user",
                "content": f"{prompt}\n\n{text}"
            }
        ]
    }
    headers = {
        "Authorization": DEFAULT.get("AUTHORIZATION_TOKEN"),
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        full_response = ""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        json_line = json.loads(line)
                        if "message" in json_line and "content" in json_line["message"]:
                            full_response += json_line["message"]["content"]
                        else:
                            logging.warning("Unexpected JSON line format: %s", json_line)
                    except json.JSONDecodeError:
                        logging.warning("非 JSON 格式的行，跳過: %s", line)
        except Exception as e:
            logging.error("讀取流式回應時發生錯誤: %s", str(e))
        return full_response
    else:
        logging.error("LLM summarization failed with status %s", response.status_code)
        raise Exception("LLM summarization failed.")


def call_llm_chat(payload: dict) -> str:
    """
    呼叫 LLM 進行對話回應。
    """
    url = DEFAULT.get("LLM_API_URL")
    headers = {
        "Authorization": DEFAULT.get("AUTHORIZATION_TOKEN"),
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    json_line = json.loads(line)
                    if 'message' in json_line and 'content' in json_line['message']:
                        full_response += json_line['message']['content']
                except json.JSONDecodeError:
                    continue
        return full_response
    else:
        error_msg = f"Request failed with status code {response.status_code}"
        logging.error(error_msg)
        raise Exception(error_msg)
