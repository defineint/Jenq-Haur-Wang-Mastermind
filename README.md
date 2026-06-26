# 正豪智多星 (Jenq-Haur-Wang Mastermind)

本專案基於王正豪教授的個人 github page 實作 Agentic Dual-Track RAG 檢索系統，專門用於解答關於 **王正豪教授 (Jenq-Haur Wang)** 的相關問題。  
本系統結合了本地端大語言模型 (Local LLM)、混合檢索 (Hybrid Search) 以及意圖路由 (Semantic Router) 技術，提供精準的問答體驗。

## 專案功能

* **Agentic 意圖路由 (Semantic Router)**：透過 Agent 分析使用者提問，自動分流至：
  * `CHITCHAT`：日常閒聊與打招呼。
  * `STRUCTURED`：結構化清單事實（學經歷、開課列表等），直接讀取本地 JSON 確保 100% 準確，**而不受到 `Top-K` 文本影響。**
  * `UNSTRUCTURED`：深層問題等細節，進入向量資料庫進行深度文獻檢索。
* **混合檢索 (Hybrid Search)**：結合 BM25 關鍵字檢索與 ChromaDB 語意向量檢索 (基於 `intfloat/multilingual-e5-large` 模型)，來解決語意權重失衡問題，並使用 RRF (倒數排名融合) 演算法提升檢索精準度。
* **自動化知識庫建置**：內建爬蟲腳本，可自動爬取教授專屬網域並轉換為乾淨的 Markdown 格式。
* **模型本地端運行**：透過 LM Studio 串接本地端 LLM，確保資料隱私與可控性 (本專案 Demo 使用 `gemma-4-E4B-it-Q4_K_M`)。
* **前端互動介面**：使用 Streamlit 提供簡潔的 Web 互動介面。

## 環境需求

* **Python 版本**：Python 3.13.0
* **硬體建議**：NVDA RTX 4060 (以加速向量資料庫建立、Embedding 計算、本地 LLM 推理)
* **本地端 LLM 伺服器**：需安裝並啟動 **LM Studio**，並將 Local Inference Server 運行於 `http://127.0.0.1:1234/v1/chat/completions`。
* **本地 LLM 模型**：`gemma-4-E4B-it-Q4_K_M` (不建議使用 Context 過短的模型，因為本專案會提取相關度前4筆的文本)

## 安裝與執行

### 1. 安裝環境與依賴套件

建議使用 Python 虛擬環境 (venv) 來執行此專案  
請依照以指令在專案根目錄 (Terminal / PowerShell) 進行操作：
(下方指令以 Windows 環境為例，不同環境請自行切換)

```bash
# 建立虛擬環境
python -m venv .venv
# 啟動 venv 環境 (Windows)
.\.venv\Scripts\activate
# 安裝所需套件
pip install .\requirements.txt
```

### 2. 資料準備與建立資料庫

依序執行以下腳本來建立知識庫（若已存在 `chroma_db_hierarchical` 且無需更新，則可略過）：

```bash
# 爬取教授網頁資料並建立層次化 JSON
python webscript.py (本程式碼將教授網頁 html file 下載下來解析，可自行調整程式改用網址)
python auto_crawler.py

# 依序執行資料清理與抽取
# python clean_data.py
# python extract_structured_facts.py

# 將清理後的資料進行 Embedding 並寫入 ChromaDB 向量資料庫
python build_hierarchical_db.py
```

### 3. 啟動系統

確保 LM Studio 的 Server 已成功開啟，接著啟動 Streamlit 前端介面：

```bash
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
```

*(註：若需透過 Cloudflare 對外開放服務，可自行部屬環境變數以及撰寫 config.yml，開啟對外網址)*

## 專案結構

* `app.py`: Streamlit 前端 Web 介面主程式。
* `agentic_rag_pipeline.py`: 專案運作核心邏輯（包含 Agent 0 語意路由、Agent 1 查詢改寫、混合檢索演算法與 Agnet 2 生成邏輯）。
* `auto_crawler.py`: 網站爬蟲腳本，負責將 HTML 結構轉為 Markdown 並存為 `hierarchical_data.json`。
* `build_hierarchical_db.py`: ChromaDB 資料庫建置腳本，負責呼叫 E5 模型並存入向量庫。
* `chroma_db_hierarchical/`: 本地端 ChromaDB 資料庫實際儲存目錄。
* `structured_facts.json`: 教授結構化事實清單，供 `STRUCTURED` 路由直接抓取回答。
* `requirements.txt`: 專案所需 Python 套件清單。

**開發者**: 黃紹泓 (雅量)
