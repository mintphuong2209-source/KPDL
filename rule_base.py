# ================= 1. FIX LỖI SQLITE (BẮT BUỘC ĐỂ TRÊN CÙNG) =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =============================================================================

import streamlit as st
import json
import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")

# 👉 TÊN FILE DỮ LIỆU CỦA BẠN (Chính xác theo ảnh bạn gửi)
JSON_FILE = "all_chunks_normalized.json" 
COLLECTION_NAME = "dichvucong_rag_v2" # Đổi tên collection để né cache cũ

# ================= 2. CẤU HÌNH API & MODEL =================
st.sidebar.header("⚙️ Cấu hình")

# Lấy Key
api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# Tự động tìm model
try:
    available_models = [m.name for m in genai.list_models() if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods]
    if available_models:
        # Ưu tiên các model ổn định
        default_idx = 0
        for i, m in enumerate(available_models):
            if "flash" in m: default_idx = i; break # Ưu tiên Flash cho nhanh
            
        SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", available_models, index=default_idx)
    else:
        st.sidebar.error("❌ Không tìm thấy model Gemini nào.")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Lỗi API: {e}")
    st.stop()

# ================= 3. HÀM LOAD DỮ LIỆU (CHỈ 1 FILE) =================
@st.cache_resource(ttl="2h") 
def initialize_vector_db():
    # Dùng model nhẹ để tiết kiệm RAM
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # Chỉ nạp nếu DB rỗng
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                st.error(f"⚠️ LỖI: Không tìm thấy file '{JSON_FILE}' trên GitHub!")
                st.info("👉 Hãy chắc chắn bạn đã upload file có tên y hệt như vậy.")
                return None
            
            with st.spinner(f"Đang đọc file {JSON_FILE}..."):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            
            if not data or not isinstance(data, list):
                st.error("❌ File dữ liệu bị rỗng hoặc sai định dạng!")
                return None
                
            total_chunks = len(data)
            st.toast(f"Đã đọc xong file. Chuẩn bị nạp {total_chunks} chunks...", icon="🚀")

            # Chuẩn bị dữ liệu
            ids = [str(uuid.uuid4()) if "id" not in item else str(item["id"]) for item in data]
            documents = [item.get("content_text", "") for item in data]
            metadatas = []
            
            for item in data:
                # Lấy metadata và xử lý None value
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                })
                clean_meta = {k: (str(v) if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # 👉 Nạp Batch size 40
            batch_size = 40
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            for i in range(0, total_chunks, batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
                progress = min((i + batch_size) / total_chunks, 1.0)
                progress_bar.progress(progress)
                status_text.caption(f"Đã nạp {int(progress*100)}%")
            
            progress_bar.empty()
            status_text.empty()
            st.toast(f"Hoàn tất! Đã nạp {total_chunks} chunks.", icon="🎉")
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
collection = initialize_vector_db()

if not collection:
    st.stop()
    
# HIỆN TRẠNG THÁI
st.sidebar.success(f"📦 Dữ liệu: **{collection.count()}** chunks")
if collection.count() < 100:
    st.sidebar.warning("⚠️ Số lượng chunk quá ít. Có thể file JSON chưa đủ dữ liệu?")

# ================= 4. LOGIC RAG =================
def query_rag(query_text, model_name, top_k=10):
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        context_parts = []
        sources = []
        
        if results["documents"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                h = meta.get('hierarchy', 'Thông tin')
                u = meta.get('url', '#')
                context_parts.append(f"--- {h} ---\n{doc}")
                sources.append(f"- [{h}]({u})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hành chính công. Hãy trả lời câu hỏi dựa trên thông tin được cung cấp.
        
        YÊU CẦU:
        - Trả lời chi tiết, rõ ràng.
        - Nếu là quy trình, hãy liệt kê từng bước.
        - Tuyệt đối không bịa đặt thông tin.
        
        NGỮ CẢNH:
        {context}
        
        CÂU HỎI: {query_text}
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, list(set(sources)), context
        
    except Exception as e:
        return f"Lỗi xử lý: {str(e)}", [], ""

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần hỗ trợ thủ tục gì?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            answer, sources, debug_ctx = query_rag(prompt, SELECTED_MODEL)
            
            if sources:
                full_resp = f"{answer}\n\n**📚 Nguồn tham khảo:**\n" + "\n".join(sources)
            else:
                full_resp = answer
            
            st.markdown(full_resp)
            
            # Debug (Tuỳ chọn)
            with st.expander("🕵️ Xem dữ liệu tìm được"):
                st.text(debug_ctx)
            
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
