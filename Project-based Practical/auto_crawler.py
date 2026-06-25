import json
import re
import time
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

INPUT_FILE = 'professor_data.json'
OUTPUT_FILE = 'hierarchical_data.json'

# 限定爬取的網域，避免爬蟲跑到外部網站(如 IEEE, ACM)
ALLOWED_DOMAINS = ['chriswjh.github.io', 'myweb.ntut.edu.tw']

# 鎖定包含課程連結的區塊
TARGET_SECTIONS = [
    "Courses Offered",
    "Information Retrieval",
    "Operating Systems",
    "Security",
    "Others: Programming"
]

def clean_markdown(text):
    # 清除連續3個以上的換行，讓文本緊湊一點
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def main():
    print("開始讀取主索引資料...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            master_chunks = json.load(f)
    except FileNotFoundError:
        print(f"錯誤: 找不到 {INPUT_FILE}")
        return

    # 1. 使用正規表示式提取 Markdown 格式的連結 [text](url)
    urls_to_crawl = set()
    url_pattern = re.compile(r'\[.*?\]\((https?://[^\)]+)\)')

    for chunk in master_chunks:
        section = chunk['metadata'].get('section', '')
        # 只挑選課程相關的區塊進行深挖
        if any(target in section for target in TARGET_SECTIONS):
            matches = url_pattern.findall(chunk['text'])
            for url in matches:
                # 過濾掉非教授網域的連結
                if any(domain in url for domain in ALLOWED_DOMAINS):
                    urls_to_crawl.add(url)

    print(f"共發現 {len(urls_to_crawl)} 個內部子網頁準備爬取。")

    # 2. 開始自動爬取並轉換
    deep_chunks = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    for url in list(urls_to_crawl):
        print(f"正在爬取: {url}")
        try:
            res = requests.get(url, headers=headers, timeout=10)
            res.raise_for_status()
            res.encoding = 'utf-8'
            
            soup = BeautifulSoup(res.text, 'lxml')
            
            # 鎖定 body 內容，略過 header 那些無用資訊
            content_html = str(soup.body) if soup.body else res.text
            
            # 將 HTML 直接無腦轉換為 Markdown
            # 忽略 script 和 style 標籤，避免抓到網頁原始碼
            markdown_text = md(content_html, heading_style="ATX", strip=['script', 'style'])
            markdown_text = clean_markdown(markdown_text)
            
            # 確保有抓到實質內容才存入
            if len(markdown_text) > 50:
                deep_chunks.append({
                    "text": markdown_text,
                    "metadata": {
                        "doc_type": "detail",        # 標記為深層細節資料
                        "parent_id": url,            # 以外部連結作為串接的 Foreign Key
                        "source_url": url
                    }
                })
            time.sleep(1) # 禮貌性延遲，避免把教授網頁打掛
        except Exception as e:
            print(f"爬取失敗 {url}: {e}")

    # 3. 合併主索引與子網頁資料
    for chunk in master_chunks:
        chunk['metadata']['doc_type'] = 'summary' # 將原本首頁的資料標記為大綱
        # 原本首頁的資料，其 parent_id 就是它自己
        chunk['metadata']['parent_id'] = chunk['metadata']['source_url']
        
    all_data = master_chunks + deep_chunks

    # 4. 儲存最終的層次化知識庫
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print(f"爬蟲完成。已合併資料並儲存至 {OUTPUT_FILE}，總資料數: {len(all_data)}。")

if __name__ == "__main__":
    main()