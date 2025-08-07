# 2025/4/29 加速版：支援重複呼叫快取 + 自動偵測模型變更
import os
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

from modules.config import DEFAULT


# --- 全域變數：管理已載入的模型資訊 ---
_current_model_info = {
    "model_name": None,
    "lora_path": None,
    "model": None,
    "tokenizer": None,
}

# def is_multimodal_template(tokenizer):
#     """判斷 tokenizer 的 chat_template 是否包含多模態格式關鍵字"""
#     template = getattr(tokenizer, "chat_template", "")
#     return any(k in template for k in ["content[0][\"type\"]", "image", "video"])


# def build_messages(user_input, tokenizer):
#     """根據 tokenizer 是否為多模態，自動建立 messages 結構"""
#     if is_multimodal_template(tokenizer):
#         return [{"role": "user", "content": [{"type": "text", "text": user_input}]}]
#     else:
#         return [{"role": "user", "content": user_input}]
    

def model_load(
    model_name: str = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length: int = 1024,
    hf_token: str = None,
    lora_path = None
) -> tuple:
    
    hf_token:str = DEFAULT.get("HF_TOKEN")
    lora_path: str = "./gemma-3b-lora"
    
    """
    載入 LLM 與 tokenizer，並套用 LoRA 權重。若已有相同設定的模型，則重用。

    參數：
        model_name (str): Hugging Face 上的模型路徑。
        max_seq_length (int): 最長輸入長度。
        hf_token (str): Hugging Face access token。
        lora_path (str): LoRA 微調權重路徑。

    回傳：
        (model, tokenizer)
    """
    global _current_model_info

    # 判斷是否需重新載入
    if (_current_model_info["model_name"] == model_name and
            _current_model_info["lora_path"] == lora_path and
            _current_model_info["model"] is not None and
            _current_model_info["tokenizer"] is not None):
        # 直接重用
        return _current_model_info["model"], _current_model_info["tokenizer"]

    # 載入 base model（不套用 LoRA）
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
        token=hf_token,
    )
    # model.eval()

    model = PeftModel.from_pretrained(model, lora_path)
    # model.eval()

    # 更新全域狀態
    _current_model_info["model_name"] = model_name
    _current_model_info["lora_path"] = lora_path
    _current_model_info["model"] = model
    _current_model_info["tokenizer"] = tokenizer

    return model, tokenizer


def call_llm_chat(payload: dict) -> str:
    """
    呼叫 LLM 進行對話回應。
    """
    model_name = payload.get("model_name", "unsloth/gemma-3-1b-it")
    lora_path = payload.get("lora_path", None)
    hf_token = payload.get("hf_token") or DEFAULT.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    max_tokens = int(payload.get("max_tokens", 512))
    model = payload.get("model", "")
    tokenizer = payload.get("tokenizer", "")
    
    if lora_path is not None:
        pass
    else:
        pass
        # 若 pad_token 未設或與 eos_token 相同，補上 pad_token
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            base = getattr(model, "base_model", model)
            base.resize_token_embeddings(len(tokenizer))

    # # 建立 messages
    # messages = build_messages(payload['messages'], tokenizer)

    # 使用 tokenizer.chat_template 產生 prompt
    prompt = tokenizer.apply_chat_template(
        payload['messages'],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
    attention_mask = attention_mask.to(model.device)

    # 推理
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.1,
            top_p=0.1,
            top_k=50,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解碼新生成部分
    prompt_len = input_ids.shape[-1]
    generated_tokens = outputs[0][prompt_len:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return answer