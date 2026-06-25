import os
import json
import torch
import jieba
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import warnings

# 隱藏不必要的警告
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
os.environ["HF_HUB_OFFLINE"] = "1"
jieba.setLogLevel(20)

DB_PATH = "./chroma_db_hierarchical"
COLLECTION_NAME = "professor_hierarchical_collection"
device_type = "cuda" if torch.cuda.is_available() else "cpu"

print("初始化系統與載入 E5 模型中...")
client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large",
    device=device_type
)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

all_data = collection.get(include=['documents', 'metadatas'])
all_docs = all_data['documents']
all_metas = all_data['metadatas']
all_ids = all_data['ids']

# 為了讓 jieba 能認得英文名字，強制加入字典
jieba.add_word("Jenq-Haur Wang")
tokenized_docs = [list(jieba.cut(doc)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)

def simulate_llm_normalization(query):
    """
    [暫時替代方案] 在還沒串接 LLM 前，使用簡單字典替換來模擬實體正規化
    這能讓 BM25 跨越中英障礙，發揮效用。
    """
    mapping = {
        "正豪": "Jenq-Haur Wang",
        "王正豪": "Jenq-Haur Wang",
        "教授": "Professor",
        "博士": "Ph.D"
    }
    normalized_query = query
    for k, v in mapping.items():
        normalized_query = normalized_query.replace(k, v)
    return normalized_query

def hybrid_search_with_zscore(query, top_k=3, rrf_k=60):
    # 1. 執行模擬的實體正規化
    norm_query = simulate_llm_normalization(query)
    print(f"\n原始查詢: '{query}'")
    print(f"正規化後: '{norm_query}' (模擬 LLM 預處理)")
    
    # 2. BM25 檢索 (使用正規化後的查詢)
    tokenized_query = list(jieba.cut(norm_query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    
    # 3. ChromaDB 檢索 (E5 本來就懂中文，所以用原查詢或正規化查詢皆可)
    e5_query = f"query: {norm_query}"
    chroma_results = collection.query(query_texts=[e5_query], n_results=len(all_docs))
    chroma_ids_sorted = chroma_results['ids'][0]
    
    # 4. RRF 融合計分
    rrf_scores = {}
    for rank, doc_idx in enumerate(bm25_ranking):
        doc_id = all_ids[doc_idx]
        rrf_scores[doc_id] = 1 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(chroma_ids_sorted):
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += 1 / (rrf_k + rank + 1)
        else:
            rrf_scores[doc_id] = 1 / (rrf_k + rank + 1)
            
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # ====================================================
    # 5. 核心：統計邊界法 (Z-Score Detection)
    # 不用通靈的 0.03，我們看前 20 名的分數分布
    # ====================================================
    top_20_scores = [score for _, score in sorted_rrf[:20]]
    mean_score = np.mean(top_20_scores)
    std_score = np.std(top_20_scores)
    
    # 計算第一名的 Z-Score (代表它比平均高出幾個標準差)
    best_score = top_20_scores[0]
    z_score = (best_score - mean_score) / std_score if std_score > 0 else 0
    
    final_results = []
    for doc_id, score in sorted_rrf[:top_k]:
        idx = all_ids.index(doc_id)
        final_results.append({
            "score": score,
            "text": all_docs[idx],
            "metadata": all_metas[idx]
        })
        
    return final_results, z_score, mean_score, std_score

def display_results(results, z_score, mean_score, std_score):
    print(f"\n統計分析數據:")
    print(f"   - 競爭圈(Top20) 平均分數: {mean_score:.4f}")
    print(f"   - 競爭圈(Top20) 標準差:   {std_score:.4f}")
    print(f"   - 第一名突出力度 (Z-Score): {z_score:.2f}")
    
    # 科學門檻：Z-Score 大於 2.0 代表結果極端突出 (在常態分布中前 2.2%)
    # 這就是你可以寫在期末報告上的「統計防線」
    Z_THRESHOLD = 2.0 
    
    if z_score < Z_THRESHOLD:
        print(f"\n[防幻覺攔截] 最高分資料未能與其他雜訊拉開差距 (Z-Score {z_score:.2f} < {Z_THRESHOLD})。系統判定為瞎猜！")
        return

    for i, res in enumerate(results):
        meta = res['metadata']
        text = res['text'].replace("passage: [王正豪 Jenq-Haur Wang] ", "")
        print(f"\n【排名 No.{i+1}】 (RRF: {res['score']:.4f})")
        print(f"類型: {meta.get('doc_type').upper()} | 來源: {meta.get('parent_id')}")
        print(f"內容: {text[:150]}...")

if __name__ == "__main__":
    print("\n統計邊界檢索系統已啟動！(純數理邏輯，未串接 LLM)")
    while True:
        user_query = input("\n請輸入測試問題 (或按 Q 退出): ")
        if user_query.lower() == 'q':
            break
        
        results, z_score, mean, std = hybrid_search_with_zscore(user_query)
        display_results(results, z_score, mean, std)