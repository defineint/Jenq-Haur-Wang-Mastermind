import json
import os
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import torch

# 1. 環境設定與硬體偵測
os.environ["HF_HUB_OFFLINE"] = "1"  # 設為離線模式，避免重複檢查模型
device_type = "cuda" if torch.cuda.is_available() else "cpu"
print(f"運算硬體偵測中... 使用: {device_type.upper()}")

# 設定路徑（使用新路徑，不覆蓋舊庫）
INPUT_FILE = 'hierarchical_data_cleaned.json'
DB_PATH = "./chroma_db_hierarchical"
COLLECTION_NAME = "professor_hierarchical_collection"

# 2. 讀取清洗後的層次化資料
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功讀取 {len(data)} 筆層次化資料。")
except FileNotFoundError:
    print(f"找不到 {INPUT_FILE}，請確認檔案路徑。")
    exit()

# 3. 初始化 ChromaDB (持久化存儲)
client = chromadb.PersistentClient(path=DB_PATH)

# 4. 載入 Embedding 模型 (multilingual-e5-large)
print(f"正在載入 Embedding 模型 (intfloat/multilingual-e5-large)...")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large",
    device=device_type
)

# 5. 建立 Collection (使用餘弦相似度)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME, 
    embedding_function=emb_fn,
    metadata={"hnsw:space": "cosine"}
)

# 清空新資料庫（確保重新跑腳本時資料不重複）
if collection.count() > 0:
    print("清除資料庫中舊有的資料...")
    existing_data = collection.get()
    collection.delete(ids=existing_data['ids'])

# 6. 資料分層處理與寫入
documents = []
metadatas = []
ids = []

print("開始進行層次化向量化存儲...")
for i, item in enumerate(data):
    # E5 模型規範：被檢索文本必須加上 "passage: " 前綴
    # 這裡我們保留你之前做的 [王正豪 Jenq-Haur Wang] 實體擴充邏輯
    prefix = "[王正豪 Jenq-Haur Wang] "
    doc_text = f"passage: {prefix}{item['text']}"
    
    documents.append(doc_text)
    metadatas.append(item['metadata'])
    ids.append(f"id_{i}")

# 因為有 RTX 4060，批次大小維持 32 以維持效能
batch_size = 32
try:
    for i in tqdm(range(0, len(documents), batch_size), desc="📥 寫入進度"):
        batch_docs = documents[i : i + batch_size]
        batch_metas = metadatas[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
    print(f"\n建立完成！層次化資料庫已存於 {DB_PATH}")
    print(f"總節點數: {collection.count()}")
except Exception as e:
    print(f"\n寫入失敗: {e}")