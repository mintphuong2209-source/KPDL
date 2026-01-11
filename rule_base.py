
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ===================================================================================

import streamlit as st
import json
import os
import glob
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Chatbot Pháp Luật", layout="wide")
st.title("🤖 Chatbot Tư Vấn Pháp Luật & Cư Trú")

COLLECTION_NAME = "dichvucong_db_final_v5" # Đổi tên mới để xóa cache cũ

# --- CẤU HÌNH API ---
st.sidebar.header("⚙️ Cấu hình")
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key.")
        st.stop()

genai.configure(api_key=api_key)

# Chọn Model
try:
    models = [m.name for m in genai.list_models() if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods]
    model_choice = st.sidebar.selectbox("Chọn Model:", models, index=0)
except:
    st.sidebar.error("Lỗi kết nối Google AI.")
    st.stop()

# ================= 3. HÀM NẠP DỮ LIỆU  =================
@st.cache_resource(ttl="2h")
def load_data_to_vector_db():
    try:
        # 1. Setup ChromaDB
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="keepitreal/vietnamese-sbert"
        )
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        # 2. Nếu DB đã có dữ liệu thì dùng luôn, không nạp lại
        if collection.count() > 0:
            return collection, f"Đã có sẵn {collection.count()} chunks."

        # 3. Tìm file JSON trong thư mục
        json_files = glob.glob("*.json")
        if not json_files:
            return None, "KHONG_TIM_THAY_FILE_JSON"

        # 4. Đọc file
        all_docs = []
        all_ids = []
        all_metas = []
        
        file_list_str = ""
        
        for file_path in json_files:
            file_list_str += f"{file_path}, "
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        content = item.get("content_text", "").strip()
                        if len(content) < 10: continue
                        
                        all_ids.append(str(uuid.uuid4()))
                        all_docs.append(content)
                        
                        # Xử lý metadata an toàn
                        meta = item.get("metadata", {}).copy()
                        meta["source_file"] = file_path
                        meta["url"] = item.get("url", "")
                        # Chroma không chịu value là None
                        clean_meta = {k: str(v) for k, v in meta.items() if v is not None}
                        all_metas.append(clean_meta)

        if not all_docs:
            return None, "FILE_RONG"

        # 5. Nạp Batch (Để không bị tràn RAM)
        batch_size = 40
        for i in range(0, len(all_ids), batch_size):
            collection.add(
                ids=all_ids[i : i+batch_size],
                documents=all_docs[i : i+batch_size],
                metadatas=all_metas[i : i+batch_size]
            )
            
        return collection, f"Thành công! Đã nạp {len(all_ids)} chunks từ: {file_list_str}"

    except Exception as e:
        return None, f"LOI_HE_THONG: {str(e)}"

# --- GỌI HÀM VÀ HIỂN THỊ KẾT QUẢ ---
with st.spinner("Đang khởi động hệ thống tri thức..."):
    collection, status_msg = load_data_to_vector_db()

# Xử lý hiển thị lỗi/thành công
if collection is None:
    if "KHONG_TIM_THAY" in status_msg:
        st.error("⚠️ Không tìm thấy file `.json` nào trên GitHub. Bạn đã upload file cào được chưa?")
    elif "LOI_HE_THONG" in status_msg:
        st.error(f"❌ {status_msg}")
        st.info("Hãy kiểm tra lại file requirements.txt xem có dòng 'pysqlite3-binary' chưa.")
    else:
        st.error(f"❌ Lỗi dữ liệu: {status_msg}")
    st.stop()
else:
    st.sidebar.success(f"✅ {status_msg}")

# ================= 4. LOGIC CHATBOT =================
def query_ai(question):
    try:
        results = collection.query(query_texts=[question], n_results=10)
        
        context = ""
        sources = []
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                source_url = meta.get('url', '#')
                file_name = meta.get('source_file', '')
                context += f"- Nội dung: {doc}\n- Nguồn: {file_name}\n---\n"
                sources.append(source_url)
        
        prompt = f"""Bạn là trợ lý ảo pháp luật. Dựa vào thông tin sau:\n{context}\n\nHãy trả lời câu hỏi: {question}"""
        
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(prompt)
        return response.text, list(set(sources))
    except Exception as e:
        return f"Lỗi khi gọi AI: {str(e)}", []

# Giao diện Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần tìm hiểu thủ tục gì?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            ans, srcs = query_ai(prompt)
            st.write(ans)
            if srcs:
                st.write("**Nguồn tham khảo:**")
                for s in srcs: st.write(f"- {s}")
            st.session_state.messages.append({"role": "assistant", "content": ans})
