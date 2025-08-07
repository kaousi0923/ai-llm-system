# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:32:54 2025

@author: scai
"""


import os
import json
import logging
import shutil
from datetime import datetime
import re

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_huggingface import HuggingFaceEmbeddings

import gc
import torch

from modules.config import EMBED_API_DEFAULTS
from typing import List


def get_embedding_function(model_name: str = None):
    """
    根據傳入的 model_name 參數建立向量化工具，
    若 model_name 為 None 則使用預設值。
    """
    if model_name is None or model_name.strip() == "":
        model_name = EMBED_API_DEFAULTS.get("MODEL")
    hugging_embed = HuggingFaceEmbeddings(model_name=model_name)
    embeddings = create_langchain_embedding(hugging_embed)
    return embeddings


def create_vector_db(name: str,
                     documents,
                     file_names: list,
                     model: str,
                     summarizer: str,
                     text_splitter_type: str,
                     chunk_size: int,
                     chunk_overlap: int,
                     vector_dbs_dir: str):
    """
    根據文件建立向量資料庫，並儲存摘要與 metadata。
    """
    vector_db_path = os.path.join(vector_dbs_dir, name)
    if os.path.exists(vector_db_path):
        raise Exception("此向量資料庫名稱已存在，請先刪除再重試。")
    os.makedirs(vector_db_path, exist_ok=True)

    # 使用 text_splitter 切割文本
    if text_splitter_type == "Recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_documents(documents)
    if not texts:
        raise Exception("文本分割結果為空，請檢查文件內容。")
    texts_content = [doc.page_content for doc in texts if hasattr(doc, "page_content")]
    if not texts_content:
        raise Exception("文本內容為空，無法創建向量資料庫。")
    embedding_model = get_embedding_function(model)
    _ = Chroma.from_documents(texts, embedding_model, persist_directory=vector_db_path)

    # 寫入摘要與 metadata
    summary_path = os.path.join(vector_db_path, "content.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summarizer)

    metadata = {
        "name": name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "summary": summarizer,
        "files": file_names
    }
    metadata_path = os.path.join(vector_db_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, ensure_ascii=False)

    logging.info(f"Vector DB created: {metadata}")
    # 釋放向量庫與模型資源
    del embedding_model
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return metadata


def list_vector_dbs(vector_dbs_dir: str, name: str = None):
    """
    列出所有或指定的向量資料庫資訊。
    """
    paths = []
    if name:
        paths = [os.path.join(vector_dbs_dir, name)]
    else:
        paths = [os.path.join(vector_dbs_dir, d) for d in os.listdir(vector_dbs_dir)]
    db_list = []
    for db_path in paths:
        if os.path.exists(db_path):
            metadata_path = os.path.join(db_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    db_list.append(metadata)
    return db_list


def delete_vector_db(name: str, vector_dbs_dir: str):
    """
    刪除指定名稱的向量資料庫。
    """
    vector_db_path = os.path.join(vector_dbs_dir, name)
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)
        logging.info(f"向量資料庫 '{name}' 已成功刪除。")
        return True
    else:
        logging.warning(f"向量資料庫 '{name}' 不存在。")
        return False


def search_vector_db(database: str, query: str, vector_dbs_dir: str, k: int = 10) -> List[str]:
    """
    支援：
    1. 精確名稱查詢，可逗號分隔多個資料庫名稱
    2. 多組 Tag 查詢（格式：(tag1, tag2), (tag3)），代表 OR 群組
    3. 單組 Tag AND 查詢（格式：(tag1, tag2)）
    """
    matching_db_paths = []

    if not database:
        return matching_db_paths

    # --- 解析輸入字串是否有多組 tag group ---
    tag_groups = re.findall(r'\((.*?)\)', database)
    name_parts = re.sub(r'\(.*?\)', '', database).split(',')  # 拿掉括號後的名稱比對部分

    # --- 處理 tag 群組邏輯 ---
    if tag_groups:
        query_tag_sets = [
            [tag.strip() for tag in group.split(',')]
            for group in tag_groups
        ]
        for d in os.listdir(vector_dbs_dir):
            db_dir = os.path.join(vector_dbs_dir, d)
            if os.path.isdir(db_dir):
                match = re.search(r'\((.*?)\)', d)
                db_tags = [tag.strip() for tag in match.group(1).split(',')] if match else []
                # 如果 db_tags 包含任一組 query_tag_set（AND 群組成立）就加入
                for tag_set in query_tag_sets:
                    if all(tag in db_tags for tag in tag_set):
                        matching_db_paths.append(db_dir)
                        break  # 一組符合就加入，跳出繼續下個資料庫

    # --- 處理資料庫名稱精確比對 ---
    for name in name_parts:
        name = name.strip()
        if name:
            db_dir = os.path.join(vector_dbs_dir, name)
            if os.path.isdir(db_dir):
                matching_db_paths.append(db_dir)

    return matching_db_paths
