import json
import re
from bs4 import BeautifulSoup, NavigableString
from urllib.parse import urljoin

BASE_URL = "https://chriswjh.github.io/"
LOCAL_FILE = "The Homepage of Jenq-Haur Wang.html"

def clean_text(text):
    """
    清除多餘的連續空白，但【保留換行符號】，
    這樣清單或表格組合在一起時，LLM 才看得懂結構。
    """
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def to_markdown(node, base_url):
    """
    將節點轉換為 Markdown。
    完美保留 [文字](連結) 格式，作為 LLM 遇到未知領域時的「退路 (Fallback)」。
    """
    if isinstance(node, NavigableString):
        return str(node)
    
    if node.name == 'a' and node.has_attr('href'):
        url = urljoin(base_url, node['href'])
        text = node.get_text(strip=True)
        return f" [{text}]({url}) "
        
    if node.name == 'br':
        return "\n" # 換行符號改回實質換行，有助於維持文字排版
        
    res = ""
    for child in node.children:
        res += to_markdown(child, base_url)
    return res

def parse_professor_html_flawless(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'lxml')
        
    extracted_chunks = []
    current_section = "General Information / 綜合資訊"
    buffer = ""

    def flush_buffer():
        nonlocal buffer
        text = clean_text(buffer)
        if len(text) > 10:
            extracted_chunks.append({
                "text": f"[{current_section}]\n{text}",
                "metadata": {
                    "section": current_section,
                    "source_url": BASE_URL 
                }
            })
        buffer = ""

    if not soup.body:
        return []

    for child in soup.body.children:
        if isinstance(child, NavigableString):
            buffer += str(child)
            
        elif child.name in ['h1', 'h2', 'h3']:
            flush_buffer()
            title = child.get_text(strip=True)
            if title:
                current_section = title
                
        # ==========================================
        # 🔥 核心修改：遇到列表時，不再切碎！
        # ==========================================
        elif child.name in ['ul', 'ol']:
            flush_buffer()
            list_content = "" # 用來收集整個清單的字串
            
            for li in child.find_all('li', recursive=False):
                li_text = clean_text(to_markdown(li, BASE_URL))
                if len(li_text) > 2:
                    # 加上 "- " 前綴並換行，模擬 Markdown 的無序清單
                    list_content += f"- {li_text}\n"
                    
            if list_content.strip():
                extracted_chunks.append({
                    # 把整個清單當作一個巨大的 Chunk 存起來
                    "text": f"[{current_section}]\n{list_content.strip()}",
                    "metadata": {"section": current_section, "source_url": BASE_URL}
                })
                
        elif child.name in ['table', 'p', 'div']:
            flush_buffer()
            block_text = clean_text(to_markdown(child, BASE_URL))
            if len(block_text) > 5 and not block_text.startswith("([NOTE]"):
                extracted_chunks.append({
                    "text": f"[{current_section}]\n{block_text}",
                    "metadata": {"section": current_section, "source_url": BASE_URL}
                })
                
        elif child.name == 'hr':
            flush_buffer()
            
        else:
            buffer += to_markdown(child, BASE_URL)
            
    flush_buffer()
    return extracted_chunks

if __name__ == "__main__":
    print("開始執行地毯式掃描解析 (大區塊反碎裂版本)...")
    data = parse_professor_html_flawless(LOCAL_FILE)
    
    with open('professor_data.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"解析完成！共萃取出 {len(data)} 個大區塊，已儲存至 professor_data.json")