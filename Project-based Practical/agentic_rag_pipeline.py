import os
import json
import torch
import jieba
import requests
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import warnings

# ==========================================
# 0. 環境與系統設定
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
os.environ["HF_HUB_OFFLINE"] = "1"
jieba.setLogLevel(20)

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
MAX_TOKENS = 2048 

DB_PATH = "./chroma_db_hierarchical"
COLLECTION_NAME = "professor_hierarchical_collection"
FACTS_FILE = "structured_facts.json" # 載入我們離線抽取的 JSON
device_type = "cuda" if torch.cuda.is_available() else "cpu"

print("[系統初始化] 載入資料庫與 E5 模型中...")
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

# 載入結構化事實資料
try:
    with open(FACTS_FILE, 'r', encoding='utf-8') as f:
        structured_facts = json.load(f)
    print(f"[系統準備就緒] 成功載入結構化事實資料包。")
except FileNotFoundError:
    structured_facts = {}
    print(f"[警告] 找不到 {FACTS_FILE}，將全程使用向量檢索。")

jieba.add_word("Jenq-Haur Wang")
tokenized_docs = [list(jieba.cut(doc)) for doc in all_docs]
bm25 = BM25Okapi(tokenized_docs)
print(f"[系統準備就緒] 成功載入 {len(all_docs)} 筆大區塊向量資料，連線 LM Studio 中...")

# ==========================================
# 核心模組：LLM API 呼叫
# ==========================================
def call_local_llm(prompt, system_prompt="", temperature=0.1):
    headers = {"Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": "local-model",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": MAX_TOKENS
    }
    try:
        response = requests.post(LM_STUDIO_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        print(f"[API 錯誤] 無法連線至 LM Studio: {e}")
        return None

# ==========================================
# Agent 0：意圖路由器 (Semantic Router) - 全新殺手鐧
# ==========================================
def route_query(user_query):
    """基於思維鏈 (CoT) 的 LLM 意圖路由器"""
    
    # 關鍵：把全域變數 structured_facts 轉成字串，準備塞給 Agent 0 當判斷依據
    json_context = json.dumps(structured_facts, ensure_ascii=False, indent=2)
    
    system_prompt = f"""
    你是一個冷靜且邏輯嚴密的意圖分析系統。請分析使用者的提問，並嚴格按照以下步驟思考：
    
    【步驟一：判斷是否為無意義閒聊】
    - 若使用者的問題毫無學術意義，只是打招呼（早安）、惡搞（吃香蕉）、或生活閒聊（天氣如何）、甚至是嘗試用正豪或是學術名詞包裝閒聊問題，請直接判定為 CHITCHAT。
    
    【步驟二：比對結構化清單 (關鍵！)】
    - 系統目前擁有的「教授結構化事實清單 (JSON)」如下：
    {json_context}
    
    - 請仔細檢查使用者的問題。
    - 如果使用者的問題「完全可以只靠」上述清單的內容回答（例如：最高學歷是什麼、有哪些工作經歷、開過哪些課），請判定為 STRUCTURED。
    - 如果清單裡「沒有」答案，或是使用者問的是課程或教授的「深層細節」（例如：OS作業系統期末考範圍、配分、論文內容、上課地點等），這代表必須去查閱外部文章，請嚴格判定為 UNSTRUCTURED。
    
    【輸出格式要求】
    1. 先寫下你的 Thought 分析過程 (警告：分析過程中絕對「不要」加上中括號)。
    2. 最後獨立換行，用中括號輸出最終結論，只能是 [CHITCHAT], [STRUCTURED] 或 [UNSTRUCTURED]。
    """
    
    print("[Agent 0 思維鏈分析] 正在對照 JSON 清單推理使用者意圖...")
    response = call_local_llm(user_query, system_prompt, temperature=0.0)
    
    if response:
        import re
        # 關鍵修正：改用 findall 抓出所有的標籤，並且只取「最後一個」！
        matches = re.findall(r'\[(CHITCHAT|STRUCTURED|UNSTRUCTURED)\]', response.upper())
        if matches:
            intent = matches[-1] # [-1] 代表永遠取 list 的最後一項
            
            # 清理一下 log，讓印出來的字串比較乾淨
            thought_process = response.replace(f'[{intent}]', '').replace('Thought:', '').strip()[:150]
            print(f"  > [推理結果] {intent} (LLM 思考過程: {thought_process}...)")
            return intent
    
    return "UNSTRUCTURED" # 預設防呆
# def route_query(user_query):
#     """判斷使用者的問題是否可以直接用結構化 JSON 回答"""
#     system_prompt = """
#     你是一個精準的意圖分類器。請分析使用者的提問，並將其歸類為以下三種標籤之一：

#     1. "STRUCTURED"：
#        - 使用者詢問教授的「學歷」、「最高學歷」或「在哪畢業」。
#        - 使用者詢問教授的「工作經歷」、「經歷」或「擔任過什麼職務」。
#        - 使用者詢問教授「開過的課程總覽」、「所有課程清單」或「特定年份開過什麼課」。

#     2. "CHITCHAT"：
#        - 使用者的輸入只是單純的打招呼（如：早安、你好、hi）。
#        - 使用者的輸入毫無意義、或是在測試系統名稱（如：正豪智多星、測試、123）。
#        - 使用者詢問與「王正豪教授的學術、課程、經歷」完全無關的問題（如：天氣如何、午餐吃什麼、講個笑話）。

#     3. "UNSTRUCTURED"：
#        - 不屬於上述兩類，且確實是在詢問教授的特定細節（如期中考時間、研究方向、論文、實驗室規定等）。

#     請只回覆 "STRUCTURED", "CHITCHAT", 或 "UNSTRUCTURED" 其中一個標籤。絕對不要輸出任何其他解釋文字。
#     """
#     print("[Agent 0 意圖路由] 正在判斷問題類型...")
#     route = call_local_llm(user_query, system_prompt, temperature=0.0)
#     # 加上一點容錯機制
#     if route:
#         # 先統一轉成大寫並去首尾空白，防範 LLM 多給空格
#         route_upper = route.upper().strip()
        
#         # 判斷是否包含 STRUCTURED 且不包含 UNSTRUCTURED
#         if "STRUCTURED" in route_upper and "UNSTRUCTURED" not in route_upper:
#             return "STRUCTURED"
#         # 判斷是否包含 CHITCHAT
#         elif "CHITCHAT" in route_upper:
#             return "CHITCHAT"
            
#     # 若都不符合，或是 route 為 None，則進入預設的 UNSTRUCTURED
#     return "UNSTRUCTURED"

# ==========================================
# Agent 1：查詢改寫器 (Query Rewriter)
# ==========================================
def rewrite_query(user_query):
    system_prompt = """
    你是一個專業的學術檢索分析員。請分析使用者的中文提問，轉換為適合搜尋引擎的英文關鍵字。
    規則 1：無論稱呼「正豪」、「王教授」，一律轉換為「Jenq-Haur Wang」。
    規則 2：精確翻譯學術名詞。
    請直接輸出英文關鍵字，不要任何解釋。
    """
    rewritten = call_local_llm(user_query, system_prompt, temperature=0.1)
    return rewritten if rewritten else user_query

# ==========================================
# 混合檢索 (Hybrid Search)
# ==========================================
def hybrid_search(original_query, rewritten_query, top_k=5, rrf_k=60):
    """大區塊版本的檢索，Top-K 不需要太大，因為每個 Chunk 都很長很完整"""
    tokenized_query = list(jieba.cut(rewritten_query))
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = np.argsort(bm25_scores)[::-1]
    
    e5_query = f"query: {original_query} {rewritten_query}"
    chroma_results = collection.query(query_texts=[e5_query], n_results=len(all_docs))
    chroma_ids_sorted = chroma_results['ids'][0]
    
    rrf_scores = {}
    for rank, doc_idx in enumerate(bm25_ranking):
        doc_id = all_ids[doc_idx]
        rrf_scores[doc_id] = 1 / (rrf_k + rank + 1)
    for rank, doc_id in enumerate(chroma_ids_sorted):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (rrf_k + rank + 1)
            
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    candidates = []
    # 因為是大區塊，稍微放寬閾值，並取前 5 筆最相關的整頁文本
    for doc_id, score in sorted_rrf[:top_k]:
        idx = all_ids.index(doc_id)
        candidates.append({
            "score": score,
            "text": all_docs[idx].replace("passage: [王正豪 Jenq-Haur Wang] ", ""),
            "meta": all_metas[idx]
        })
    return candidates

# ==========================================
# Agent 2：嚴格裁判與生成器 (Generator)
# ==========================================
def generate_final_answer(user_query, context_text, route_type):
    system_prompt = """
    你是一位專業的學術助理。請嚴格根據以下【參考資料】來回答問題，絕對不要使用外部知識。
    
    回答守則：
    1. 使用者提問中的「正豪」、「王教授」，指的就是「Jenq-Haur Wang」。
    2. 誠實作答：若資料中確實找不到解答，請回答：「根據檢索到的資料，並未提及相關資訊。」
    3. 精準溯源：只要是從網頁文本提取的資訊，請在結尾獨立一行附上「來源網址」。(若資料是來自結構化清單，則無需附網址，直接列出結果即可)。
    4. 請使用繁體中文，語氣保持專業與流暢。
    """

    full_prompt = f"【使用者提問】\n{user_query}\n\n【參考資料】\n{context_text}\n\n請根據上述資料回答。"
    
    print("\n" + "▼"*20 + f" [DEBUG: {route_type} 參考資料] " + "▼"*20)
    print(context_text[:1500] + "...\n(以下省略)" if len(context_text) > 1500 else context_text)
    print("▲"*60 + "\n")

    print("[生成中] 正在交由 LLM 進行最終組織與生成...")
    final_answer = call_local_llm(full_prompt, system_prompt, temperature=0.2)
    return final_answer

# ==========================================
# 主程式執行迴圈
# ==========================================
# if __name__ == "__main__":
#     print("\n" + "="*60)
#     print("正豪智多星 (Agentic Dual-Track RAG) 終極進化版上線！")
#     print("="*60)
    
#     while True:
#         query = input("\n請問有什麼我可以幫忙的？ (輸入 Q 退出): ")
#         if query.lower() == 'q':
#             break
            
#         # 步驟 1：意圖路由
#         route = route_query(query)
        
#         context_for_llm = ""
#         if route == "STRUCTURED":
#             print("[路由判定] 觸發「結構化事實大腦 (JSON)」！繞過向量庫，保證 100% 精準。")
#             # 將 JSON 轉為 LLM 容易閱讀的字串格式
#             context_for_llm = "【王正豪教授結構化事實清單】\n" + json.dumps(structured_facts, ensure_ascii=False, indent=2)
            
#         else:
#             print("[路由判定] 觸發「語意化大腦 (ChromaDB)」！進入深度檢索...")
#             rewritten = rewrite_query(query)
#             print(f"  > [關鍵字擴充] {rewritten}")
            
#             candidates = hybrid_search(query, rewritten, top_k=4) # 取最相關的 4 個大網頁區塊
            
#             for i, doc in enumerate(candidates):
#                 url = doc['meta'].get('source_url', 'unknown')
#                 context_for_llm += f"\n--- 資料 {i+1} (來源: {url}) ---\n{doc['text']}\n"

#         # 步驟 2：生成最終答案
#         answer = generate_final_answer(query, context_for_llm, route)
        
#         print("\n【AI 助理回覆】")
#         print(answer)
#         print("-" * 60)


# 將這段放在 agentic_rag_pipeline.py 的最下方
def get_agent_response(query):
    """給前端呼叫的 API 介面"""
    route = route_query(query)

    context_for_llm = ""
    if route == "CHITCHAT":
        
        # 1. 直接定義好要回傳的 chitchat 答案
        answer = "你好，我是正豪智多星!  \n我專門回答關於 王正豪教授 (Jenq-Haur Wang) 的學歷、經歷、歷年開課資訊等問題。  \n請問有什麼我能幫您的嗎？"
        
        return answer, route
    elif route == "STRUCTURED":
        context_for_llm = "【王正豪教授結構化事實清單】\n" + json.dumps(structured_facts, ensure_ascii=False, indent=2)
    else:
        rewritten = rewrite_query(query)
        candidates = hybrid_search(query, rewritten, top_k=4)
        for i, doc in enumerate(candidates):
            url = doc['meta'].get('source_url', 'unknown')
            context_for_llm += f"\n--- 資料 {i+1} (來源: {url}) ---\n{doc['text']}\n"

    # 生成答案
    answer = generate_final_answer(query, context_for_llm, route)
    print("回傳答案!!!")
    # 回傳答案，也順便回傳 route，讓前端可以炫耀系統用了哪個大腦！
    return answer, route