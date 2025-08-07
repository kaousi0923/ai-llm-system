 

from langchain_chroma import Chroma  # 將向量儲存在向量資料庫
from langchain_core.document_loaders.base import Document
from embedding_function import get_embedding_function
import pandas as pd

# 定義路徑
CHROMA_PATH = "../chroma"
FILE_PATH = "../AI_output2.xlsx"

# 使用 pandas 讀取 Excel 檔案
dataframe = pd.read_excel(FILE_PATH)

# 確保資料框包含所需的欄位，例如 "IndexName" 和 "Type"
if "Q" not in dataframe.columns or "A" not in dataframe.columns:
    raise ValueError("Excel 文件必須包含 'Q' 和 'A' 欄位！")

# 將資料轉換為 langchain 的 Document 格式
data = [
    Document(page_content=str(row.dropna().to_dict()), metadata={"Q": row["Q"], "A": row["A"]})
    for _, row in dataframe.iterrows()
]

# 確認資料是否正確載入
print(f"Loaded {len(data)} documents")
print(data[0])  # 測試顯示第一筆資料

# 初始化向量資料庫
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=get_embedding_function()
)

# 為每筆文件生成唯一 ID
uuids = [str(i) for i in range(len(data))]

# 新增文件到向量資料庫
db.add_documents(documents=data, ids=uuids)

print("Done")
