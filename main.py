import os

os.environ["PYTHONUTF8"] = "1"
os.environ["DISABLE_TORCH_INDUCTOR"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE"] = "0"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import gc
import re
import shutil
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo # 時間套件
from typing import Union  # 最上面記得加

# 清除暫存： 
temp_path = os.path.expandvars(r"%LOCALAPPDATA%\Temp\torchinductor_scai")
shutil.rmtree(temp_path, ignore_errors=True)

import uvicorn
import fastapi_cdn_host
from fastapi.responses import JSONResponse, StreamingResponse
import requests
from io import BytesIO



from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from typing import List, Optional

from modules.config import DEFAULT, EMBED_API_DEFAULTS, CHAT_API_DEFAULTS
from modules import file_processing, vector_db, chat_history, llm_caller_unsloth

from opencc import OpenCC
# 微調用
import pandas as pd
import torch
from datasets import Dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
from openai import OpenAI

from langchain_community.vectorstores import Chroma
#底層需安裝pip install rank_bm25
from langchain_community.retrievers import BM25Retriever
#中文分詞器 # pip install jieba
import jieba 


from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

    
# 全域變數，紀錄當前使用中的模型與 tokenizer
current_model = None
current_tokenizer = None
#current_lora = None
current_model_name = None

# 從環境變數或配置文件取得 OpenAI API Key
openai_api_key = DEFAULT.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("警告: 未設置 OPENAI_API_KEY，OpenAI 功能將無法使用")
    client = None
else:
    client = OpenAI(api_key=openai_api_key)

# 訓練後模型存放目錄
TRAINED_MODELS_DIR = DEFAULT.get("TRAINED_MODELS_DIR", "trained_models")
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


# 基底模型存放目錄
BASE_MODELS_DIR = DEFAULT.get("BASE_MODELS_DIR", "trained_models")
os.makedirs(BASE_MODELS_DIR, exist_ok=True)
os.environ["BASE_MODELS_DIR"] = BASE_MODELS_DIR



# 自訂轉換函式：處理字串轉換為 int 或 float，若為空字串則回傳預設值
def parse_optional_int(value: Optional[str], default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

def parse_optional_float(value: Optional[str], default: float) -> float:
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default

# 設定日誌
logs_dir = DEFAULT.get("LOGS_DIR")
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, f"api_{datetime.now().strftime('%Y-%m-%d')}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

app = FastAPI(
    title='GAI訓練平台-後端測試文件',
    description='\n',
    version='0.5.3'
)
fastapi_cdn_host.patch_docs(app, favicon_url='/img/MIRDC_Logo.png')

# 加入預設基底模型清單（靜態）
payment_models = DEFAULT.get("payment_model")
payment_models = [bm.strip() for bm in payment_models.split(",") if bm.strip()]
hf_token = DEFAULT.get("HUGGINGFACE_TOKEN") or DEFAULT.get("HF_TOKEN")
    
    
    
def unload_current_model():
    """卸載並釋放目前模型的 GPU 記憶體。"""
    global current_model, current_tokenizer, current_model_name

    if current_model is not None:
        # 1. 把模型搬回 CPU（可避免 GPU 上殘留參數）
        try:
            current_model.cpu()
        except Exception:
            pass

        # 2. 刪除所有參考
        del current_model
        del current_tokenizer
        # if current_lora is not None:
        #     del current_lora

        # 3. Python 垃圾回收 + PyTorch 清除快取
        gc.collect()
        torch.cuda.empty_cache()

    # 重置全域指標
    current_model = None
    current_tokenizer = None
    # current_lora = None
    current_model_name=None
    
    
    
# -----------------------------------
# 建立向量資料庫 API (POST /api/embed)
# 輸入：
#  • name (必填)：建立的向量資料庫名稱，例如：公務車(汽油)
#  • files (必填)：上傳的檔案，支持上傳多個檔案，至少支援 PDF、docx、txt 檔案
#  • model (選填)：設置用於轉換向量的模型 (預設：None)
#  • summarizer (選填)：上傳資料描述或總結；若傳入文本則直接作為總結，若留空則由 LLM 自動摘要
#  • text_splitter (選填)：文本分割器選擇，預設為 Recursive
#  • chunk_size (選填)：文本分割器每段 Tokens 上限，預設 1500
#  • chunk_overlap (選填)：文本分割器段落間銜接 Tokens 數，預設 50
#  • summary_threshold (選填)：當 summarizer 為空時，用於控制 LLM 一次讀取文本量的上限，預設 100000
# 輸出：
#  • summary (字串)：經 LLM 總結後的文本描述
# -----------------------------------
@app.post("/api/embed", tags=["RAG操作"], summary='建立向量資料庫')
async def create_vector_db(
    name: str = Form(..., description="建立的向量資料庫名稱，例如：公務車(汽油)"),
    files: List[UploadFile] = File(..., description="上傳的檔案，支持上傳多個檔案，至少支援 PDF、docx、txt 檔案"),
    model: Optional[str] = Form(None, description="設置用於轉換向量的模型 (預設：None)"),
    summarizer: Optional[str] = Form(None, description="上傳資料描述或總結；若傳入文本則直接作為總結，若留空則由 LLM 自動摘要"),
    text_splitter: Optional[str] = Form(None, description="文本分割器選擇，預設為 Recursive"),
    chunk_size: Optional[str] = Form(None, description="文本分割器每段 Tokens 上限，預設 1500"),
    chunk_overlap: Optional[str] = Form(None, description="文本分割器段落間銜接 Tokens 數，預設 50"),
    summary_threshold: Optional[str] = Form(None, description="當 summarizer 為空時，用於控制 LLM 一次讀取文本量的上限，預設 100000")
):
    vector_dbs_dir = DEFAULT.get("VECTOR_DBS_DIR")
    os.makedirs(vector_dbs_dir, exist_ok=True)

    # 讀取 EMBED API 預設參數
    embed_default_model = EMBED_API_DEFAULTS.get("MODEL")
    embed_default_summarizer = EMBED_API_DEFAULTS.get("SUMMARIZER")
    embed_default_text_splitter = EMBED_API_DEFAULTS.get("TEXT_SPLITTER")
    embed_default_chunk_size = int(EMBED_API_DEFAULTS.get("CHUNK_SIZE"))
    embed_default_chunk_overlap = int(EMBED_API_DEFAULTS.get("CHUNK_OVERLAP"))
    embed_default_summary_threshold = int(EMBED_API_DEFAULTS.get("SUMMARY_THRESHOLD"))

    # 使用傳入參數或預設值，並轉換數值型參數
    model = model or embed_default_model
    summarizer = summarizer if summarizer is not None else embed_default_summarizer
    text_splitter_type = text_splitter or embed_default_text_splitter
    chunk_size_val = parse_optional_int(chunk_size, embed_default_chunk_size)
    chunk_overlap_val = parse_optional_int(chunk_overlap, embed_default_chunk_overlap)
    summary_threshold_val = parse_optional_int(summary_threshold, embed_default_summary_threshold)

    # 提取名稱與標籤 (若有)
    clean_name, tags = file_processing.extract_tags(name)
    final_name = f"{clean_name}({','.join(tags)})" if tags else clean_name

    vector_db_path = os.path.join(vector_dbs_dir, final_name)
    if os.path.exists(vector_db_path):
        raise HTTPException(
            status_code=400,
            detail=("此向量資料庫名稱已被使用，請使用 /api/embed/delete 功能刪除後重新呼叫")
        )

    temp_dir = os.path.join(DEFAULT.get("TEMP_DIR"))
    os.makedirs(temp_dir, exist_ok=True)

    try:
        documents, file_names = file_processing.process_uploaded_files(files, temp_dir)
        final_summary = summarizer

        vector_db.create_vector_db(
            name=final_name,
            documents=documents,
            file_names=file_names,
            model=model,
            summarizer=final_summary,
            text_splitter_type=text_splitter_type,
            chunk_size=chunk_size_val,
            chunk_overlap=chunk_overlap_val,
            vector_dbs_dir=vector_dbs_dir
        )

        # 使用 OpenCC 轉換為臺灣繁體中文
        cc = OpenCC("s2twp")
        converted_summary = cc.convert(final_summary)
        metadata_path = os.path.join(vector_db_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            metadata["summary"] = converted_summary
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False)

        logging.info(f"Vector DB created: {metadata}")
        return {"summary": converted_summary}
    except Exception as e:
        logging.error(f"Error creating vector DB: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def is_multimodal_template(tokenizer):
    """
    判斷 tokenizer 的 chat_template 是否需要多模態格式。
    """
    template = getattr(tokenizer, "chat_template", "")
    return any(k in template for k in ["content[0][\"type\"]", "image", "video"])

def normalize_messages(messages, tokenizer):
    """
    根據 tokenizer 的需求，將 messages 中的 content 欄位轉換為適當的格式。
    """
    if is_multimodal_template(tokenizer):
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
    return messages

# -----------------------------------
# 推理/聊天功能 API (POST /api/chat)
# 輸入：
#  • model (選填)：選擇用於推理的模型名稱，例如：unsloth/gemma-3-4b-it
#  • messages (必填)：傳入的聊天訊息 (純字串)
#  • database (選填)：呼叫指定的向量資料庫 (可複選)，預設為 None
#  • template (選填)：聊天文字模板，預設為 None
#  • temperature (選填)：控制 LLM 回答隨機性，通常 0~2，預設 0
#  • top_p (選填)：控制回應用詞精準度，通常 0~1，預設 0
#  • max_tokens (選填)：控制回應最大 Tokens 數量，預設 2048
#  • history_length (選填)：限制歷史聊天輪數，預設 3
#  • retrieval_k (選填)：向量相似度門檻，預設 0.5
#  • user_id (選填)：使用者識別碼，預設為 None
# 輸出：
#  • content (字串)：LLM 回覆的內容
# -----------------------------------
def load_model_if_needed(model: str):
    global current_model, current_tokenizer, current_model_name

    if model in payment_models:
        current_model = None
        current_tokenizer = None
        current_model_name = model
        return

    if current_model_name != model:
        unload_current_model()
        current_model, current_tokenizer = FastLanguageModel.from_pretrained(
            model_name=model,
            max_seq_length=DEFAULT.get("MAX_SEQ_LENGTH", 2048),
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            token=hf_token,
            load_in_4bit=True
        )
        current_model_name = model


def extract_named_entities_with_llm(user_query: str, model: str):
    ner_prompt = f"""請從以下問題中提取重要的命名實體（如人名、地名、組織名、產品名、規章名、時間等），並改寫成常見的用詞 例如"GRI202之2"，改寫成"GRI 202-2"，以逗號分隔的形式返回：

問題：{user_query}

請只返回提取的實體，用逗號分隔，不要其他解釋："""

    ner_messages = [
        {"role": "system", "content": "你是一個專業的命名實體提取助手，請準確提取文本中的關鍵實體。"},
        {"role": "user", "content": ner_prompt}
    ]

    try:
        load_model_if_needed(model)
        if model in payment_models:
            response = client.chat.completions.create(
                model=model,
                messages=ner_messages,
                temperature=0.1,
                top_p=0.1,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
        else:
            payload = {
                "model_name": model,
                "hf_token": hf_token,
                "messages": ner_messages,
                "top_p": 0.1,
                "max_tokens": 200,
                "temperature": 0.1,
                "model": current_model,
                "tokenizer": current_tokenizer
            }
            content = llm_caller_unsloth.call_llm_chat(payload)

        return [e.strip() for e in content.split(",") if len(e.strip()) > 1]
    except Exception as e:
        logging.error(f"命名實體提取失敗: {str(e)}")
        return []
# 問題轉寫功能
def rewrite_query_to_statement(user_query: str, model: str):
    """將問題轉寫成敘述句，用於語意搜尋"""
    rewrite_prompt = f"""請將以下問題轉寫成一個完整的敘述句，用於文件搜尋。轉寫時請：
1. 保留問題的核心關鍵詞
2. 將疑問句轉換為陳述句
3. 使問題更明確和具體
4. 保持原意不變

原問題：{user_query}

請只返回轉寫後的敘述句，不要其他解釋："""

    rewrite_messages = [
        {"role": "system", "content": "你是一個專業的問題轉寫助手，請準確將問題轉寫成適合搜尋的敘述句。"},
        {"role": "user", "content": rewrite_prompt}
    ]

    try:
        load_model_if_needed(model)
        if model in payment_models:
            response = client.chat.completions.create(
                model=model,
                messages=rewrite_messages,
                temperature=0.1,
                top_p=0.1,
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
        else:
            payload = {
                "model_name": model,
                "hf_token": hf_token,
                "messages": rewrite_messages,
                "top_p": 0.1,
                "max_tokens": 200,
                "temperature": 0.1,
                "model": current_model,
                "tokenizer": current_tokenizer
            }
            content = llm_caller_unsloth.call_llm_chat(payload)

        return content
    except Exception as e:
        logging.error(f"問題轉寫失敗: {str(e)}")
        return user_query  # 如果轉寫失敗，返回原問題


# 混合搜索功能
def hybrid_search(user_query: str, model: str, matching_db_paths: list, embeddings, k: int = 10):
    """混合搜索：結合命名實體提取關鍵字搜尋和語意搜尋"""

    
    all_documents = []
    all_chunks = []
    
    # 1. 命名實體提取
    extracted_entities = extract_named_entities_with_llm(user_query, model)
    logging.info(f"自動提取的命名實體: {extracted_entities}")
    
    # 2. 問題轉寫
    rewritten_query = rewrite_query_to_statement(user_query, model)
    logging.info(f"轉寫後的查詢: {rewritten_query}")
    
    # 3. 從所有資料庫收集文件
    for db_path in matching_db_paths:
        try:
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            # 取出原始文件
            raw_docs = vectorstore.get()["documents"]
            documents = [Document(page_content=doc) for doc in raw_docs]
            all_documents.extend(documents)
            
        except Exception as e:
            logging.error(f"Error loading database at {db_path}: {str(e)}")
    
    if not all_documents:
        return []
    
    # 4. 文件切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    all_chunks = splitter.split_documents(all_documents)
    
    # 5. 建立檢索器
    def chinese_tokenizer(text):
        return list(jieba.cut(text))
    
    try:
        # BM25檢索器（關鍵字搜尋）
        bm25_retriever = BM25Retriever.from_documents(
            all_chunks,
            tokenizer=chinese_tokenizer
        )
        bm25_retriever.k = k
        
        # 向量檢索器（語意搜尋）
        # 重新建立向量資料庫用於語意搜尋
        temp_vectorstore = Chroma.from_documents(
            all_chunks,
            embedding=embeddings
        )
        vector_retriever = temp_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # 混合檢索器
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.6, 0.4]  # BM25和向量搜尋各佔50%權重
        )
        
        # 6. 執行混合搜索
        final_results = []
        
        # 使用原問題進行混合搜索
        hybrid_docs = ensemble_retriever.get_relevant_documents(user_query)
        final_results.extend(hybrid_docs)
        
        # 使用轉寫後的問題進行額外的語意搜尋
        if rewritten_query != user_query:
            semantic_docs = vector_retriever.get_relevant_documents(rewritten_query)
            final_results.extend(semantic_docs)
        
        # 使用提取的實體進行關鍵字搜尋
        if extracted_entities:
            for entity in extracted_entities:
                entity_docs = bm25_retriever.get_relevant_documents(entity)
                final_results.extend(entity_docs)
        
        # 去重並限制結果數量
        unique_docs = []
        seen_content = set()
        for doc in final_results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                if len(unique_docs) >= k * 2:  # 限制最終結果數量
                    break
        
        logging.info(f"混合搜索結果: 找到 {len(unique_docs)} 個相關文件")
        return unique_docs
        
    except Exception as e:
        logging.error(f"混合搜索失敗: {str(e)}")
        return []
#將 RAG + 圖片包裝成 統一內部格式 
def build_multimodal_prompt(retrieved_contexts: str, user_query: str) -> str:
    return f"以下是從知識庫檢索到的內容：\n{retrieved_contexts}\n\n使用者的問題如下：\n{user_query}"
def build_user_message_for_model(prompt_text: str, image_content: Optional[dict], payment_model: bool):
    if payment_model:
        # OpenAI 多模態格式
        content = [{"type": "text", "text": prompt_text}]
        if image_content:
            content.append(image_content)
        return {"role": "user", "content": content}
    else:
        # 本地模型格式（純文字）
        return {"role": "user", "content": prompt_text}
    
# 將 RAG + 圖片包裝成統一內部格式 
def build_multimodal_prompt(retrieved_contexts: str, user_query: str) -> str:
    return f"以下是從知識庫檢索到的內容：\n{retrieved_contexts}\n\n使用者的問題如下：\n{user_query}"

def build_user_message_for_model(prompt_text: str, image_content: Optional[dict], payment_model: bool):
    if payment_model:
        # OpenAI 多模態格式
        content = [{"type": "text", "text": prompt_text}]
        if image_content:
            content.append(image_content)
        return {"role": "user", "content": content}
    else:
        # 本地模型格式（純文字）
        return {"role": "user", "content": prompt_text}
    
@app.post("/api/chat", tags=["對話功能"], summary='對話聊天功能')
async def chat_with_vector_db(
    model: Optional[str] = Form(..., description="""選擇用於推理的模型名稱，例如：
                                \n gpt-4o-mini    (OpenAI)
                                \n gpt-4.1    (OpenAI)
                                \n chatgpt-4o-latest    (OpenAI)
                                \n unsloth/gemma-3-4b-it-unsloth-bnb-4bit    (本地初始模型)
                                \n unsloth/gemma-3-1b-it-unsloth-bnb-4bit    (本地初始模型)
                                \n 數位島機器人4b    (本地微調模型)"""),
    messages: str = Form(..., description="傳入的聊天訊息 (純字串)"),
    database: Optional[str] = Form(None, description="呼叫指定的向量資料庫 (可複選)，預設為 None"),
    template: Optional[str] = Form(None, description="聊天文字模板，不指定則預設模板"),
    temperature: Optional[str] = Form(None, description="控制 LLM 回答隨機性 (0~2)，預設 0"),
    top_p: Optional[str] = Form(None, description="控制回應用詞精準度 (0~1)，預設 0"),
    max_tokens: Optional[str] = Form(None, description="控制回應最大 Tokens 數量，預設 2048"),
    history_length: Optional[str] = Form(None, description="限制歷史聊天輪數，預設 3"),
    retrieval_k: Optional[str] = Form(None, description="向量相似度門檻，預設 0.5"),
    user_id: Optional[str] = Form(None, description="使用者識別碼，預設為 None"),
    # 新增圖片上傳相關參數
    image: Union[UploadFile, str, None] = File(None, description="上傳的圖片檔案 (僅限 OpenAI 多模態模型使用)"),
    enable_vision: Optional[bool] = Form(False, description="是否啟用視覺功能 (僅限 OpenAI 多模態模型)"),
):  
    import base64
    from io import BytesIO
    
    global current_model, current_tokenizer, current_model_name
    
    logging.info(f"Chat request received: model={model}, database={database}, enable_vision={enable_vision}")
   
    
    # 讀取 CHAT API 預設參數
    chat_default_model = CHAT_API_DEFAULTS.get("MODEL")
    chat_default_template = CHAT_API_DEFAULTS.get("TEMPLATE")
    chat_default_temperature = float(CHAT_API_DEFAULTS.get("TEMPERATURE"))
    chat_default_top_p = float(CHAT_API_DEFAULTS.get("TOP_P"))
    chat_default_max_tokens = int(CHAT_API_DEFAULTS.get("MAX_TOKENS"))
    chat_default_history_length = int(CHAT_API_DEFAULTS.get("HISTORY_LENGTH"))
    chat_default_retrieval_k = float(CHAT_API_DEFAULTS.get("RETRIEVAL_K"))

    model = model or chat_default_model
    template = template or chat_default_template or DEFAULT.get("DEFAULT_PROMPT_TEMPLATE")
    temperature_val = parse_optional_float(temperature, chat_default_temperature)
    top_p_val = parse_optional_float(top_p, chat_default_top_p)
    max_tokens_val = parse_optional_int(max_tokens, chat_default_max_tokens)
    history_length_val = parse_optional_int(history_length, chat_default_history_length)
    retrieval_k_val = parse_optional_float(retrieval_k, chat_default_retrieval_k)
    
    # 🔍 動態讀取模型詳細資訊 (從list_models讀取)
    matched_model = next((m for m in list_models(model) if m.get("name") == model), None)
    base_model = matched_model.get("base_model") if matched_model else model
    lora_name = matched_model.get("lora_name") if matched_model else None
    
    PAYMENT_FLAG = False
    
    # 檢查是否為付費模型
    for openai_model in payment_models:
        if openai_model == model:
            PAYMENT_FLAG = True
            break
    
    # 處理圖片上傳相關邏輯
    image_content = None
    
    # 修正：先檢查圖片是否為空或無效
    if image is not None and hasattr(image, 'content_type'):
        # 檢查是否啟用視覺功能
        if enable_vision:
            if not PAYMENT_FLAG:
                raise HTTPException(
                    status_code=400, 
                    detail="圖片上傳功能僅支援 OpenAI 付費模型"
                )
        
            # 檢查檔案類型
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail="上傳的檔案必須是圖片格式"
                )
        
            try:
                image_bytes = await image.read()
                if len(image_bytes) == 0:
                    logging.warning("上傳的圖片檔案為空")
                    image_content = None
                    enable_vision = False
                else:
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.content_type};base64,{image_base64}"
                        }
                    }
                    logging.info(f"圖片上傳成功，檔案大小: {len(image_bytes)} bytes")
            except Exception as e:
                logging.error(f"圖片處理失敗: {str(e)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"圖片處理失敗: {str(e)}"
                )
        else:
            # 如果有圖片但未啟用視覺功能，發出警告
            logging.warning("檢測到圖片上傳但未啟用視覺功能，將忽略圖片")
    else:
        # 沒有圖片或圖片無效
        if enable_vision:
            logging.warning("啟用視覺功能但未提供有效圖片")
            enable_vision = False
    
    # 本地模型不支援圖片的檢查
    if not PAYMENT_FLAG and image_content:
        raise HTTPException(
            status_code=400, 
            detail="本地模型不支援圖片上傳功能，請使用 OpenAI 付費模型"
        )
    
    user_query = messages
    chat_history_list = []
    
    # 載入聊天歷史（如果有user_id）
    if user_id:
        chat_history_list = chat_history.load_history(user_id)

    # RAG檢索邏輯
    vector_dbs_dir = DEFAULT.get("VECTOR_DBS_DIR")
    matching_db_paths = vector_db.search_vector_db(database, user_query, vector_dbs_dir)
    retrieved_contexts = ""
    
    # 如果有資料庫，使用混合搜索
    if matching_db_paths:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_API_DEFAULTS.get("MODEL"))
        
        # 使用混合搜索
        retrieved_docs = hybrid_search(
            user_query=user_query,
            model=model,
            matching_db_paths=matching_db_paths,
            embeddings=embeddings,
            k=int(retrieval_k_val * 10)  # 增加搜索結果數量
        )
        
        if retrieved_docs:
            retrieved_contexts = "\n\n".join([doc.page_content for doc in retrieved_docs])
            logging.info(f"混合搜索成功，共找到 {len(retrieved_docs)} 個相關文件")
        else:
            retrieved_contexts = "無相關資訊"
            logging.warning("混合搜索未找到相關文件")
    else:
        retrieved_contexts = "無相關資訊"
        logging.info("未指定資料庫，跳過搜索")

    # 構建訊息邏輯
    if user_id:  # 有使用者ID，需要記錄對話歷史
        if not chat_history_list:  # 第一次對話
            # 建立多模態內容
            if PAYMENT_FLAG:
                # OpenAI 多模態格式
                multimodal_content = [{
                    "type": "text",
                    "text": f"{template}\n以下是從知識庫檢索到的內容：\n{retrieved_contexts}\n\n使用者問題如下：\n{user_query}"
                }]
                
                if image_content:
                    multimodal_content.append(image_content)
                
                new_user_message = {
                    "role": "user",
                    "content": multimodal_content
                }
            else:
                # 本地模型純文字格式
                content_text = f"{template}\n以下是從知識庫檢索到的內容：\n{retrieved_contexts}\n\n使用者問題如下：\n{user_query}"
                new_user_message = {
                    "role": "user",
                    "content": content_text
                }
            
            chat_history_list.append(new_user_message)
            
        else:  # 不是第一次對話
            if PAYMENT_FLAG:
                # OpenAI 多模態格式
                multimodal_content = [{"type": "text", "text": user_query}]
                if image_content:
                    multimodal_content.append(image_content)
                
                new_user_message = {
                    "role": "user",
                    "content": multimodal_content
                }
            else:
                # 本地模型純文字格式
                new_user_message = {"role": "user", "content": user_query}
            
            chat_history_list.append(new_user_message)
            
            # 檢查是否超過對話輪次上限
            if len(chat_history_list) > history_length_val * 2:  # *2 因為包含user和assistant
                chat_history_list = chat_history_list[-(history_length_val * 2):]
        
        # 系統訊息
        system_message = {
            "role": "system",
            "content": "你是一個專業的助理，根據提供的查詢資訊用繁體中文回答使用者問題。如果使用者提供了圖片，請詳細分析圖片內容並結合文字問題給出回答。"
        }
        
        payload_messages = [system_message] + chat_history_list

    else:  # 無使用者ID，不記錄對話歷史
        # RAG prompt 組合
        prompt_text = build_multimodal_prompt(retrieved_contexts, user_query)
        
        # 根據模型選擇 content 格式
        user_message = build_user_message_for_model(prompt_text, image_content, PAYMENT_FLAG)
        
        # 系統指令
        system_message = {
            "role": "system",
            "content": "你是一個專業的助理，請根據知識庫資訊與圖片內容回答問題，並使用繁體中文。"
        }
        
        # 組合對話
        payload_messages = [system_message, user_message]
    
    # 模型推理邏輯
    if PAYMENT_FLAG:  # 付費模型 (OpenAI)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=payload_messages,
                temperature=temperature_val,
                top_p=top_p_val,
                max_tokens=max_tokens_val
            )
            full_response = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI API 呼叫失敗: {str(e)}")
            raise HTTPException(status_code=500, detail=f"模型回應失敗: {str(e)}")
            
    else:  # 本地模型
        try:
            if current_model_name == model:  # 相同模型，重用
                logging.info('使用相同模型，沿用已載入的模型')
                
                # 若是多模態格式需要轉換
                if is_multimodal_template(current_tokenizer):
                    for msg in payload_messages:
                        if isinstance(msg.get("content"), str):
                            msg["content"] = [{"type": "text", "text": msg["content"]}]
                
                # 組合 payload
                payload = {
                    "model_name": base_model,
                    "hf_token": hf_token,
                    "messages": payload_messages,
                    "top_p": top_p_val,
                    "max_tokens": max_tokens_val,
                    "temperature": temperature_val,
                    "model": current_model,
                    "tokenizer": current_tokenizer
                }
                
                if lora_name is not None:
                    payload["lora_path"] = os.path.join(TRAINED_MODELS_DIR, lora_name)
                    
            else:  # 不同模型，需要重新載入
                logging.info('載入新模型')
                unload_current_model()  # 卸載現有模型
                
                # 載入新模型
                current_model, current_tokenizer = FastLanguageModel.from_pretrained(
                    model_name=base_model,
                    max_seq_length=DEFAULT.get("MAX_SEQ_LENGTH", 2048),
                    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    token=hf_token,
                    load_in_4bit=True
                )
                
                # 若是多模態格式需要轉換
                if is_multimodal_template(current_tokenizer):
                    for msg in payload_messages:
                        if isinstance(msg.get("content"), str):
                            msg["content"] = [{"type": "text", "text": msg["content"]}]
                
                # 載入 LoRA 權重（如果有）
                if lora_name is not None:
                    from peft import PeftModel
                    current_model = PeftModel.from_pretrained(current_model, os.path.join(TRAINED_MODELS_DIR, lora_name))
                
                # 組合 payload
                payload = {
                    "model_name": base_model,
                    "hf_token": hf_token,
                    "messages": payload_messages,
                    "top_p": top_p_val,
                    "max_tokens": max_tokens_val,
                    "temperature": temperature_val,
                    "model": current_model,
                    "tokenizer": current_tokenizer
                }
                
                if lora_name is not None:
                    payload["lora_path"] = os.path.join(TRAINED_MODELS_DIR, lora_name)
                
                current_model_name = model  # 更新全局模型名稱
            
            # 呼叫本地模型
            full_response = llm_caller_unsloth.call_llm_chat(payload)
            
        except Exception as e:
            logging.error(f"本地模型推理失敗: {str(e)}")
            raise HTTPException(status_code=500, detail=f"本地模型推理失敗: {str(e)}")

    # 保存對話歷史（如果有user_id）
    if user_id:
        assistant_message = {"role": "assistant", "content": full_response}
        chat_history_list.append(assistant_message)
        chat_history.save_history(user_id, chat_history_list)
    
    logging.info(f"Chat response generated. User ID: {user_id}, Vision enabled: {enable_vision}")
    return {"content": full_response}


def update_base_models_config(new_base_model: str):
    import configparser
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH, encoding="utf-8")
    existing = config.get("DEFAULT", "BASE_MODELS", fallback="")
    model_list = [m.strip() for m in existing.split(",") if m.strip()]
    if new_base_model not in model_list:
        model_list.append(new_base_model)
        config.set("DEFAULT", "BASE_MODELS", ",".join(model_list))
        with open(CONFIG_FILE_PATH, "w", encoding="utf-8") as f:
            config.write(f)
            
            
@app.post(
    "/api/train",
    tags=["模型訓練/微調"],
    summary="微調模型功能 (LoRA)"
)
async def train_model(
    model: str = Form(..., description="基底模型名稱，例如：unsloth/gemma-3-1b-it-unsloth-bnb-4bit"),
    new_model: str = Form(..., description="訓練後模型名稱，用於呼叫/識別，例如：數位島機器人v1"),
    file: UploadFile = File(..., description="上傳規範化後的 .xlsx 檔案，須含 Q/A 欄位"),
    method: Optional[str] = Form("LoRA", description="訓練方法 (LoRA)"),
    learning_rate: Optional[float] = Form(1e-5, description="學習率 (通常 1e-5 ~ 1e-4)"),
    batch_size: Optional[int] = Form(16, description="每批樣本數 (通常 16 ~ 128)"),
    epochs: Optional[int] = Form(5, description="迭代次數 (通常 3 ~ 5)"),
    optimizer: Optional[str] = Form("AdamW", description="優化器 (AdamW/SGD/LAMB)"),
    r: Optional[int] = Form(64, description="LoRA 降維矩陣的秩 (通常 4 ~ 64)"),
    dropout: Optional[float] = Form(0.05, description="LoRA 適配層 dropout 比例 (0 ~ 0.1)")
):
    # 1. 保存上傳檔案到暫存目錄
    temp_dir = os.path.join(DEFAULT.get("TEMP_DIR", "./temp"), new_model)
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 2. 載入並檢查資料集
    df = pd.read_excel(file_path, engine="openpyxl")
    if not all(col in df.columns for col in ["Q", "A"]):
        raise HTTPException(status_code=400, detail="Excel 檔案必須包含 'Q' 和 'A' 欄位")
    
    # 移除缺值並轉為字串
    df = df[["Q", "A"]].dropna()
    instruction_list = [str(q) for q in df["Q"]]
    output_list = [str(a) for a in df["A"]]
    
    dataset = Dataset.from_dict({
        "instruction": instruction_list,
        "output": output_list
    })

    # 3. 準備模型與 tokenizer
    base_model_name = model
    update_base_models_config(base_model_name)
    
    hf_token = DEFAULT.get("HF_TOKEN")
    if not hf_token:
        raise HTTPException(status_code=500, detail="缺少 HF_TOKEN 設定，無法執行微調")
    max_seq_length = int(DEFAULT.get("MAX_SEQ_LENGTH", 2048))

    llm_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True,
        token=hf_token,
    )

    # 4. 加掛 LoRA 模組 (僅支援 LoRA)
    if method.upper() == "LORA":
        llm_model = FastLanguageModel.get_peft_model(
            llm_model,
            r=r,
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ],
            lora_alpha=r,
            lora_dropout=dropout,
            bias="none",
            use_rslora=False,
            modules_to_save=["embed_tokens", "lm_head"],
        )
    else:
        raise HTTPException(status_code=400, detail="目前僅支援 LoRA 方法")

    # 5. 設定訓練參數
    output_dir = os.path.join(TRAINED_MODELS_DIR, new_model)
    os.makedirs(output_dir, exist_ok=True)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        optim=optimizer.lower() + "_torch",
        save_steps=100,
        save_strategy="no",
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.0,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        disable_tqdm=False,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=llm_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        formatting_func=lambda ex: tokenizer.apply_chat_template([
            [{"role": "user", "content": i}, {"role": "assistant", "content": o}]
            for i, o in zip(ex["instruction"], ex["output"])
        ], tokenize=False),
        packing=False,
        dataset_num_proc=1,
        args=training_args,
    )

    # 6. 執行訓練並保存模型
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    trainer.tokenizer.save_pretrained(output_dir)
    # 7. 建立並儲存 metadata
    created_at = datetime.now(tz=ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
    metadata = {
        "name": new_model,
        "base_model": base_model_name,
        "lora_name": new_model,
        "created_at": created_at,
        "files": [file.filename]
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)

    # 清理暫存檔案
    shutil.rmtree(temp_dir, ignore_errors=True)

    return {"status": "success", "model_name": new_model}


@app.get(
    "/api/model/list",
    tags=["對話功能"],
    summary="查詢已訓練模型與基底模型列表"
)
def list_models(
    model_name: Optional[str] = Query(None, description="指定查詢特定模型，若不提供則回傳所有")
):
    models = []

    # 加入微調模型
    for folder in os.listdir(TRAINED_MODELS_DIR):
        meta_file = os.path.join(TRAINED_MODELS_DIR, folder, "meta.json")
        if os.path.isfile(meta_file):
            with open(meta_file, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            if model_name and meta.get("name") != model_name:
                continue
            models.append(meta)

    # 加入預設基底模型清單（靜態）
    base_models_str = DEFAULT.get("BASE_MODELS", "")
    base_models = [bm.strip() for bm in base_models_str.split(",") if bm.strip()]

    for bm in base_models:
        if model_name and bm != model_name:
            continue
        models.append({
            "name": bm,
            "base_model": bm,
            "lora_name": None,
            "created_at": None,
            "files": None
        })

    return models

# -----------------------------------
# 查詢向量資料庫列表 API (GET /api/embed/list)
# 輸入：
#  • name (選填)：指定查詢特定資料庫，若不提供則回傳所有資料庫
# 輸出：
#  • name：資料庫名稱
#  • created_at：資料庫創建時間（以太平洋時間 UTC+8）
#  • summary：資料庫概述
#  • files：該資料庫建立使用的檔案名稱
# -----------------------------------
@app.get("/api/embed/list", tags=["RAG操作"], summary='查詢向量資料庫')
def list_vector_dbs(name: Optional[str] = Query(None, description="指定查詢特定資料庫，若不提供則回傳所有資料庫")):
    vector_dbs_dir = DEFAULT.get("VECTOR_DBS_DIR")
    db_list = vector_db.list_vector_dbs(vector_dbs_dir, name)
    return db_list

# -----------------------------------
# 刪除向量資料庫 API (DELETE /api/embed/delete)
# 輸入：
#  • name (必填)：要刪除的資料庫名稱
# 輸出：
#  • status：狀態碼 (success, error)
#  • message：回傳訊息
# -----------------------------------
@app.delete("/api/embed/delete", tags=["RAG操作"], summary='刪除向量資料庫')
def delete_vector_db(name: str = Query(..., description="要刪除的資料庫名稱")):
    vector_dbs_dir = DEFAULT.get("VECTOR_DBS_DIR")
    try:
        if vector_db.delete_vector_db(name, vector_dbs_dir):
            return {"status": "success", "message": f"向量資料庫 '{name}' 已刪除。"}
        else:
            return {"status": "error", "message": f"向量資料庫 '{name}' 不存在。"}
    except Exception as e:
        logging.error(f"刪除向量資料庫 '{name}' 時發生錯誤: {str(e)}")
        return {"status": "error", "message": f"刪除向量資料庫時發生錯誤: {str(e)}"}


SUPPORTED_MODELS = ["gpt-4o-mini", "gpt-4.1"]

DEFAULT_PROMPT = (
    "請將問題「{question}」用不同角度重新表述 {expand_count} 種方式，"
    "同時將回答「{answer}」轉換成較口語化的說法，各 {expand_count} 種。"
    "請使用以下格式輸出（請勿使用陣列或巢狀結構）：\n"
    "{\n"
    "  \"Q_expansion_1\": \"...\",\n"
    "  \"Q_expansion_2\": \"...\",\n"
    "  ...\n"
    "  \"A_expansion_1\": \"...\",\n"
    "  \"A_expansion_2\": \"...\"\n"
    "}\n"
    "請勿包含多餘文字或解釋說明。"
)

def process_with_logging(df: pd.DataFrame, model_name: str, api_key: str, expand_count: int = 5, temperature: float = 0.5, custom_prompt: str = None) -> tuple[str, str]:
    client = OpenAI(api_key=api_key)
    results = pd.DataFrame(columns=["IndexName", "Type", "Q", "A"])
    error_rows = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = f"log_{timestamp}.txt"

    with open(log_file_path, "w", encoding="utf-8") as logf:
        logf.write(f"=== 文本擴增 LOG 開始時間 {timestamp} ===\n\n")

        for idx, row in df.iterrows():
            Q, A = str(row.get("Q", "")), str(row.get("A", ""))
            seq = idx + 1

            example_json = "{\n"
            for i in range(1, expand_count + 1):
                example_json += f"  \"Q_expansion_{i}\": \"...\",\n"
            for i in range(1, expand_count + 1):
                comma = "," if i != expand_count else ""
                example_json += f"  \"A_expansion_{i}\": \"...\"{comma}\n"
            example_json += "}"

            if custom_prompt:
                prompt = custom_prompt.replace("{question}", Q).replace("{answer}", A).replace("{expand_count}", str(expand_count))
            else:
                prompt = (
                    f"請假設自己是一個碳盤查系統的愚蠢使用者，"
                    f"將下列原始問題改寫為 {expand_count} 個不同但相關、具多樣詞彙的繁體中文問題(請分別以Q_expansion_1 ~ Q_expansion_{expand_count}表示)，"
                    f"並使用同義詞或換句話說的方式進行修飾。"
                    f"另外，請將下列答案改寫成更口語化或更生動的繁體中文回答，同樣產生 {expand_count} 個回答，"
                    f"並分別以A_expansion_1 ~ A_expansion_{expand_count}表示。\n\n"
                    f"原始問題: {Q}\n原始答案: {A}\n\n"
                    f"請以JSON格式輸出，結構如下：\n{example_json}"
                )

            generated_text = None
            for _ in range(3):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "你是一個將LLM文本資料擴增的機器人，請務必嚴格遵守使用者指示，以指定的JSON格式輸出。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        top_p=1
                    )
                    generated_text = response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    logf.write(f"[ERROR] 第 {seq} 筆 GPT 呼叫失敗：{repr(e)}\n")

            results = pd.concat([results, pd.DataFrame([{
                "IndexName": f"R{seq}_1", "Type": "R", "Q": Q, "A": A
            }])], ignore_index=True)

            if not generated_text:
                logf.write(f"[FAIL] 第 {seq} 筆 完全未能生成\n\n")
                for i in range(1, expand_count + 1):
                    results = pd.concat([results, pd.DataFrame([{
                        "IndexName": f"G{seq}_{i}", "Type": "G", "Q": "", "A": ""
                    }])], ignore_index=True)
                error_rows.append(seq)
                continue

            try:
                data = json.loads(generated_text)
                formatted_data = {}
                if all(k in data for k in ["questions", "answers"]):
                    for i in range(expand_count):
                        q_item = data["questions"][i] if i < len(data["questions"]) else ""
                        a_item = data["answers"][i] if i < len(data["answers"]) else ""
                        formatted_data[f"Q_expansion_{i+1}"] = q_item["text"] if isinstance(q_item, dict) else q_item
                        formatted_data[f"A_expansion_{i+1}"] = a_item["text"] if isinstance(a_item, dict) else a_item
                else:
                    formatted_data = data

                for i in range(1, expand_count + 1):
                    gen_Q = formatted_data.get(f"Q_expansion_{i}", "")
                    gen_A = formatted_data.get(f"A_expansion_{i}", "")
                    if not gen_Q or not gen_A:
                        error_rows.append(seq)
                    results = pd.concat([results, pd.DataFrame([{
                        "IndexName": f"G{seq}_{i}", "Type": "G", "Q": gen_Q, "A": gen_A
                    }])], ignore_index=True)
            except json.JSONDecodeError:
                logf.write(f"[FAIL] 第 {seq} 筆 JSON解析失敗\n內容：{generated_text}\n\n")
                error_rows.append(seq)
                for i in range(1, expand_count + 1):
                    results = pd.concat([results, pd.DataFrame([{
                        "IndexName": f"G{seq}_{i}", "Type": "G", "Q": "", "A": ""
                    }])], ignore_index=True)

        logf.write("\n=== 總結 ===\n")
        if error_rows:
            logf.write(f"⚠️ 擴增失敗筆數：{len(set(error_rows))}\n")
            logf.write(f"⚠️ 錯誤行列索引：{sorted(set(error_rows))}\n")
        else:
            logf.write("✅ 所有資料成功擴增！\n")

    output_path = f"output_{timestamp}.xlsx"
    results.to_excel(output_path, index=False)
    return output_path, log_file_path
# 
@app.post(
    "/api/data/augment",
    tags=["擴充功能"],
    summary="問答優化/擴增"
)
async def augment(
    file: UploadFile = File(..., description="請上傳 Excel 檔案（需包含「Q」與「A」欄位）"),
    api_key: str = Form(None, description="請輸入 OpenAI API 金鑰"),
    model_name: str = Form("gpt-4o-mini", description="指定使用的模型（預設為 gpt-4o-mini）"),
    expand_count: int = Form(5, description="欲擴增的問題與答案數（預設為 5）"),
    temperature: float = Form(0.5, description="模型創意溫度值（0~2，預設為 0.5）較高的值會讓產生的資料更多變化"),
    custom_prompt: str = Form(
    default=DEFAULT_PROMPT,
    description=(
        "（可選）自訂提示詞。"
        "請務必保留 {question}、{answer}、{expand_count} 三個變數，"
        "並指示 LLM 嚴格輸出符合系統格式的 JSON 結構："
        "Q_expansion_n / A_expansion_n。"
    ),
    example=DEFAULT_PROMPT),
):
    if model_name not in SUPPORTED_MODELS:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": f"不支援的模型名稱: {model_name}"
        })
    if not api_key:
        api_key = openai_api_key or DEFAULT.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


    try:
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        output_file, log_file = process_with_logging(df, model_name, api_key, expand_count, temperature, custom_prompt)
        # 將 Excel 讀入記憶體
        with open(output_file, "rb") as f:
            excel_bytes = BytesIO(f.read())

        # 清除暫存檔案（如需保留可略過）
        os.remove(output_file)

        return StreamingResponse(
            excel_bytes,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(output_file)}"
            }
        )
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == '__main__':
    try:
        import tunnel_setup
        tunnel_setup.main()
    except ImportError:
        logging.warning("tunnel_setup 模組不存在，略過外網穿透設定")
    port = os.getenv('TUNNEL_PORT', '5000')
    uvicorn.run(app, host="0.0.0.0", port=int(port))
