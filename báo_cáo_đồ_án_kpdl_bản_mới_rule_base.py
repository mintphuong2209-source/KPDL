# ================= 1. FIX LỖI SQLITE (BẮT BUỘC) =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ===============================================================

import streamlit as st
import json
import os
import glob
import time
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from google.api_core import exceptions

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Chatbot Pháp Luật", layout="wide")
st.title("🤖 Chatbot Tư Vấn (Chế độ Fallback Thông Minh)")

COLLECTION_NAME = "dichvucong_smart_v1"

# ================= 2. CẤU HÌNH API =================
st.sidebar.header("⚙️ Cấu hình")
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key.")
        st.stop()

genai.configure(api_key=api_key)

# --- DANH SÁCH MODEL ƯU TIÊN (Theo thứ tự) ---
# Hệ thống sẽ thử lần lượt từ trên xuống dưới
PRIORITY_MODELS = [
    "gemini-1.5-flash",          # Ưu tiên 1: Nhanh, rẻ
    "gemini-1.5-pro",            # Ưu tiên 2: Thông minh hơn (nếu Flash lỗi)
    "gemini-1.0-pro",            # Ưu tiên 3: Bản cũ ổn định
    "gemini-1.5-flash-latest"    # Ưu tiên 4: Bản mới nhất
]

# ================= 3. HÀM GỌI AI THÔNG MINH (FALLBACK + RETRY) =================
def call_smart_ai(prompt):
    """
    Hàm này tự động thử các model khác nhau cho đến khi thành công.
    """
    debug_logs = []
    
    for model_name in PRIORITY_MODELS:
        # Với mỗi model, thử tối đa 2 lần nếu mạng chập chờn
        for attempt in range(2):
            try:
                # Tạo model
                model = genai.GenerativeModel(model_name)
                
                # Gọi AI
                response = model.generate_content(prompt)
                
                # Nếu thành công -> Trả về ngay
                return response.text, f"✅ Đã trả lời bằng: **{model_name}**"
                
            except Exception as e:
                error_msg = str(e)
                # Nếu lỗi quota (hết tiền/hết lượt) -> Bỏ qua model này ngay lập tức
                if "429" in error_msg or "ResourceExhausted" in error_msg:
                    debug_logs.append(f"⚠️ {model_name}: Quá tải (Quota exceeded).")
                    break # Thoát vòng lặp retry, chuyển sang model kế tiếp
                
                # Nếu lỗi khác -> Thử lại 1 lần nữa
                debug_logs.append(f"⚠️ {model_name} (Lần {attempt+1}): {error_msg}")
                time.sleep(1) # Nghỉ 1s
    
    # Nếu thử hết tất cả mà vẫn tạch
    return None, "\n".join(debug_logs)

# ================= 4. HÀM NẠP DỮ LIỆU =================
@st.cache_resource(ttl="2h")
def load_database():
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="keepitreal/vietnamese-sbert"
        )
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        if collection.count() == 0:
            json_files = glob.glob("*.json")
            if not json_files: return None, "NO_FILES"

            all_ids, all_docs, all_metas = [], [], []
            for file_path in json_files:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            content = item.get("content_text", "").strip()
                            if len(content) < 10: continue
                            all_ids.append(str(uuid.uuid4()))
                            all_docs.append(content)
                            meta = item.get("metadata", {}).copy()
                            meta["url"] = item.get("url", "#")
                            clean_meta = {k: str(v) for k, v in meta.items() if v is not None}
                            all_metas.append(clean_meta)

            if not all_docs: return None, "EMPTY"

            batch_size = 40
            for i in range(0, len(all_ids), batch_size):
                collection.add(
                    ids=all_ids[i:i+batch_size],
                    documents=all_docs[i:i+batch_size],
                    metadatas=all_metas[i:i+batch_size]
                )
        return collection, "OK"
    except Exception as e:
        return None, str(e)

# --- KHỞI ĐỘNG ---
with st.spinner("Đang khởi động hệ thống..."):
    collection, status = load_database()

if collection:
    st.sidebar.success(f"✅ Dữ liệu sẵn sàng: {collection.count()} chunks")
else:
    st.error(f"Lỗi: {status}")
    st.stop()

# ================= 5. XỬ LÝ CÂU HỎI =================
def query_system(question):
    # 1. Tìm kiếm dữ liệu
    results = collection.query(query_texts=[question], n_results=5)
    
    context = ""
    links = []
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            url = meta.get('url', '#')
            context += f"- {doc}\n"
            if url != '#': links.append(url)
    
    if not context:
        return "Không tìm thấy thông tin trong dữ liệu.", "", []

    # 2. Tạo Prompt
    prompt = f"""
    Bạn là trợ lý ảo pháp luật. Dựa vào thông tin sau:\n{context}
    \nHãy trả lời câu hỏi: {question}
    """
    
    # 3. Gọi hàm thông minh (Fallback)
    answer, log_info = call_smart_ai(prompt)
    
    if answer:
        return answer, log_info, list(set(links))
    else:
        return f"Xin lỗi, tất cả các hệ thống AI đều đang bận.\nChi tiết lỗi:\n{log_info}", "", []

# ================= 6. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần hỏi gì?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            ans, debug_info, sources = query_system(prompt)
            
            st.markdown(ans)
            
            # Hiển thị model nào đã trả lời (để bạn biết)
            if debug_info:
                st.caption(debug_info)
                
            if sources:
                st.markdown("**Nguồn tham khảo:**")
                for s in sources: st.markdown(f"- {s}")
            
            st.session_state.messages.append({"role": "assistant", "content": ans})
