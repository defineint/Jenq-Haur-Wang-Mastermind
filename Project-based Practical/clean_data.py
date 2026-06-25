import json
import re
import requests
import urllib3
from requests.exceptions import RequestException, SSLError, ReadTimeout

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

INPUT_FILE = 'hierarchical_data.json'
OUTPUT_FILE = 'hierarchical_data_cleaned.json'

# 🔥 新增：已知死亡的舊網域，直接判死刑，不浪費時間等待 Timeout
KNOWN_DEAD_DOMAINS = ['myweb.ntut.edu.tw']

def verify_url_robust(url):
    # 1. 遇到已知死連結，直接秒殺拔除
    if any(domain in url for domain in KNOWN_DEAD_DOMAINS):
        return False
        
    internal_domains = ["chriswjh.github.io", "ntut.edu.tw"]
    
    # 2. 外部連結無條件放行
    is_internal = any(domain in url for domain in internal_domains)
    if not is_internal:
        return True

    # 3. 內部活著的網域，進行輕量級 HTTP 檢查
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        # 將 timeout 降到 3 秒，加快處理速度
        response = requests.head(url, headers=headers, timeout=3, allow_redirects=True, verify=False)
        
        if response.status_code in [401, 403, 405, 418]:
            return True
            
        if response.status_code == 404 or response.status_code >= 500:
            print(f"  [內部大掃除] 清除本站死連結 ({response.status_code}): {url}")
            return False
            
        return True
        
    except (SSLError, ReadTimeout):
        return True
    except RequestException as e:
        print(f"  [內部大掃除] 內部網域無法連線，拔除連結: {url}")
        return False

def clean_detail_content(text):
    nav_pattern = r'\[\s*(Home|Research|Publication|Teaching|Laboratory).*?\]\(.*?\)'
    text = re.sub(nav_pattern, '', text, flags=re.IGNORECASE)
    text = re.sub(r'-{3,}', '---', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.split('Start of StatCounter Code')[0]
    return text.strip()

def process_summary_text(text):
    urls = re.findall(r'\[(.*?)\]\((https?://.*?)\)', text)
    processed_text = text
    
    for link_text, url in urls:
        if not verify_url_robust(url):
            # 完美支援大區塊：只拔除網址，保留課程/論文名稱文字！
            processed_text = processed_text.replace(f"[{link_text}]({url})", link_text)
    
    garbage_keywords = ["More on the details", "For more information", "You can also view the list"]
    if len(processed_text) < 150 and any(kw in processed_text for kw in garbage_keywords):
        return None
        
    return processed_text

def main():
    print("開始執行「防禦性動態連線檢查 (大區塊加速版)」...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_data = []
    dropped_count = 0
    
    for item in data:
        text = item['text']
        metadata = item['metadata']
        
        if metadata.get('doc_type') == 'detail':
            cleaned_text = clean_detail_content(text)
            item['text'] = f"Detailed content of {metadata['source_url']}:\n{cleaned_text}"
            if len(cleaned_text) > 100:
                cleaned_data.append(item)
            else:
                dropped_count += 1
                
        else: # Summary 大區塊
            final_text = process_summary_text(text)
            if final_text is None:
                dropped_count += 1
                continue
                
            item['text'] = final_text
            metadata['parent_id'] = "MASTER_INDEX"
            cleaned_data.append(item)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

    print(f"\n清洗完成！")
    print(f"原始資料: {len(data)} 筆")
    print(f"成功物理刪除無價值垃圾節點: {dropped_count} 筆")
    print(f"剩餘高質量資料: {len(cleaned_data)} 筆")
    print(f"已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()