import json
import requests
import re

# 你的本地 LLM API (Gemma-4-e4b)
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
INPUT_FILE = 'professor_data.json'
OUTPUT_FILE = 'structured_facts.json'

def call_llm_for_json(prompt, system_prompt):
    """呼叫 LLM 並強制其吐出 JSON 格式"""
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1, # 溫度調到極低，確保它乖乖做事實萃取，不亂發揮
        "max_tokens": 2048
    }
    
    try:
        response = requests.post(LM_STUDIO_URL, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content'].strip()
        
        # 簡單清理 Markdown 的 ```json ... ``` 標籤
        content = re.sub(r'^```json\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
        
        return json.loads(content)
    except Exception as e:
        print(f"[LLM 萃取失敗]: {e}")
        return None

def main():
    print("開始執行 [結構化事實離線萃取]...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 準備一個空的字典，用來裝教授的所有結構化資料
    professor_facts = {
        "name": "Jenq-Haur Wang (王正豪)",
        "highest_degree": "",
        "education": [],
        "experience": [],
        "courses_taught": []
    }

    # 1. 處理學歷 (Education)
    edu_chunk = next((item for item in data if "Education" in item['metadata']['section']), None)
    if edu_chunk:
        print("正在萃取學歷資訊...")
        sys_prompt = "你是一個精確的資料萃取 AI。請閱讀提供的文本，將學歷資訊轉換為嚴格的 JSON 格式，必須包含 'highest_degree' (最高學歷，如 Ph.D 或 BS) 與 'history' (陣列，包含 degree, school, year)。不要輸出任何說明文字。"
        result = call_llm_for_json(edu_chunk['text'], sys_prompt)
        if result:
            professor_facts['highest_degree'] = result.get('highest_degree', '')
            professor_facts['education'] = result.get('history', [])

    # 2. 處理經歷 (Experience)
    exp_chunk = next((item for item in data if "Experience" in item['metadata']['section']), None)
    if exp_chunk:
        print("正在萃取經歷資訊...")
        sys_prompt = "你是一個精確的資料萃取 AI。請閱讀提供的文本，將工作經歷轉換為嚴格的 JSON 陣列，每個元素包含 'title' (職稱), 'organization' (機構), 'period' (期間)。請直接輸出 JSON 陣列，不要輸出任何說明文字。"
        result = call_llm_for_json(exp_chunk['text'], sys_prompt)
        if result:
            professor_facts['experience'] = result

    # 3. 處理課程清單 (將所有跟課程有關的區塊合併起來給 LLM)
    course_sections = [
        "Courses Offered", 
        "Information Retrieval/Natural Language Processing", 
        "Operating Systems", 
        "Security", 
        "Others: Programming"
    ]
    course_text = "\n".join([item['text'] for item in data if any(s in item['metadata']['section'] for s in course_sections)])
    
    if course_text:
        print("正在萃取歷年開課清單 (這可能需要一點時間)...")
        sys_prompt = """
        你是一個精確的資料萃取 AI。請閱讀提供的文本，列出這位教授『開過的所有課程名稱』，並且【務必附上該課程曾開課的年份或學期】。
        請直接輸出一個 JSON 陣列，格式範例如下：
        [
          {"course": "Operating Systems", "years": ["Spring 2020", "Fall 2021", "Spring 2022"]},
          {"course": "Data Mining", "years": ["Fall 2019"]}
        ]
        不要輸出任何說明文字，只要純 JSON 陣列。
        """
        result = call_llm_for_json(course_text, sys_prompt)
        if result:
            professor_facts['courses_taught'] = result

    # 儲存結果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(professor_facts, f, ensure_ascii=False, indent=4)
        
    print(f"\n萃取完成！完美的結構化資料已儲存至 {OUTPUT_FILE}。")
    print(json.dumps(professor_facts, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()