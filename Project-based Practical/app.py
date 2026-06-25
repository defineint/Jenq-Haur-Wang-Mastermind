# streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
# cloudflared tunnel --config config.yml run NLP-final-demo

import warnings
import logging
import os

# 1. 忽略 Python 的 Deprecation 和 User 警告
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYSLOG_WARNING"] = "1"

import streamlit as st
from agentic_rag_pipeline import get_agent_response

# 1. 設定頁面主題與標題
st.set_page_config(page_title="正豪智多星", layout="centered")

# === 初始化 Session State 狀態 ===
if "searching" not in st.session_state:
    st.session_state.searching = False
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "final_route" not in st.session_state:
    st.session_state.final_route = None

# 2. 標題區塊
st.title("正豪智多星~")
st.subheader("Agentic Dual-Track RAG 檢索系統")
st.write("歡迎提問！本系統將自動分析您的意圖，選擇「結構化事實」或「語意化文獻」為您解答。")

st.markdown("---")

# 3. 互動區塊：使用者提問框
query = st.text_input(
    "請輸入您的問題：", 
    placeholder="例如：正豪的最高學歷是什麼？ 或 2025年NLP期中考時間？",
    disabled=st.session_state.searching
)

# 4. 按鈕與執行邏輯
def click_search_button():
    if query:
        st.session_state.searching = True
        # 點擊新問題時，先清空上一次的答案
        st.session_state.final_answer = None
        st.session_state.final_route = None
    else:
        st.warning("請先輸入問題喔！")

search_button = st.button(
    "開始檢索", 
    type="primary", 
    disabled=st.session_state.searching,
    on_click=click_search_button
)

# 當狀態進入 searching = True 時，開始執行後端 Agent 檢索
if st.session_state.searching:
    with st.spinner("智多星思考中，正在調度 Agent..."):
        try:
            answer, route = get_agent_response(query)
            
            # 關鍵：把答案存進 session_state，這樣 rerun 後才不會消失
            st.session_state.final_answer = answer
            st.session_state.final_route = route
            
        except Exception as e:
            st.error(f"系統發生錯誤：{str(e)}")
            
        finally:
            # 執行完畢，解鎖按鈕並強制刷新頁面
            st.session_state.searching = False
            st.rerun()

# === 答案渲染區塊 ===
# 無論是剛跑完，還是頁面刷新後，只要狀態裡有答案，就持續顯示在畫面上
if st.session_state.final_answer and st.session_state.final_route:
    route = st.session_state.final_route
    answer = st.session_state.final_answer
    
    # 依據路由顯示不同的徽章
    if route == "CHITCHAT":
        st.info("路由判定：偵測為「日常閒聊/打招呼」，直接由系統回覆。")
    elif route == "STRUCTURED":
        st.info("路由判定：觸發「結構化事實大腦 (JSON)」，確保 100% 事實精準。")
    else:
        st.info("路由判定：觸發「語意化大腦 (ChromaDB)」，進行深度文獻檢索。")
    
    st.markdown("### 回覆結果")
    st.success(answer)