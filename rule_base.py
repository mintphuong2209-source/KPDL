# ================= 1. FIX LỖI SQLITE (BẮT BUỘC ĐỂ TRÊN CÙNG) =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =============================================================================

import streamlit as st
import json
import os
import glob
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")
COLLECTION_NAME = "dichvucong_rag"

# ================= 2. XỬ LÝ API KEY & CHỌN MODEL (AUTO) =================
st.sidebar.header("⚙️ Cấu hình")

# Lấy Key từ Secrets hoặc nhập tay
api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# Tự động tìm model Gemini khả dụng để tránh lỗi 404
try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            if 'gemini' in m.name:
                available_models.append(m.name)
    
    if available_models:
        # Chọn model đầu tiên tìm thấy (thường là Flash hoặc Pro)
        SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", available_models, index=0)
    else:
        st.sidebar.error("❌ Không tìm thấy model Gemini nào. Kiểm tra lại Key.")
        st.stop()
        
except Exception as e:
    st.sidebar.error(f"Lỗi kết nối API: {e}")
    st.stop()

# ================= 3. HÀM LOAD DỮ LIỆU TỰ ĐỘNG (NHIỀU FILE) =================
@st.cache_resource(ttl="2h") # Tự xóa cache sau 2h để cập nhật dữ liệu mới
def initialize_vector_db():
    # 👉 DÙNG MODEL NHẸ ĐỂ KHÔNG BỊ TRÀN RAM
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
            
            # Tự động tìm tất cả file .json trong thư mục
            list_json_files = glob.glob("*.json")
            
            if not list_json_files:
                st.error("⚠️ Không tìm thấy file .json nào trên GitHub!")
                return None
                
            st.toast(f"Tìm thấy {len(list_json_files)} file dữ liệu. Đang đọc...", icon="📂")
            
            all_data = []
            
            # Đọc và gộp tất cả file JSON
            for file_name in list_json_files:
                with open(file_name, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                    except:
                        pass # Bỏ qua file lỗi
            
            if not all_data:
                st.error("Dữ liệu rỗng!")
                return None

            # Chuẩn bị dữ liệu
            ids = [str(uuid.uuid4()) if "id" not in item else item["id"] for item in all_data]
            documents = [item["content_text"] for item in all_data]
            metadatas = []
            
            for item in all_data:
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                })
                # Xóa giá trị None
                clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # 👉 Nạp Batch size 40 (An toàn cho RAM)
            batch_size = 40
            progress_bar = st.sidebar.progress(0)
            status_text = st.sidebar.empty()
            
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
                progress = min((i + batch_size) / len(ids), 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Đang nạp: {int(progress*100)}%")
            
            progress_bar.empty()
            status_text.empty()
            st.toast(f"Đã nạp xong {len(ids)} chunks!", icon="✅")
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
collection = initialize_vector_db()

if not collection:
    st.stop()
    
# Hiện số lượng chunk
st.sidebar.success(f"📦 Dữ liệu: **{collection.count()}** chunks")

# ================= 4. LOGIC RAG (TOP_K = 12) =================
def query_rag(query_text, model_name, top_k=12):
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
                hierarchy = meta.get('hierarchy', meta.get('title', 'Thông tin'))
                url = meta.get('url', '#')
                context_parts.append(f"--- MỤC: {hierarchy} ---\n{doc}")
                sources.append(f"- [{hierarchy}]({url})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hành chính công. Trả lời câu hỏi dựa trên thông tin sau.
        
        YÊU CẦU:
        - Trả lời chi tiết, rõ ràng, từng bước.
        - Nếu không có thông tin, hãy nói "Chưa tìm thấy thông tin trong dữ liệu".
        
        NGỮ CẢNH:
        {context}
        
        CÂU HỎI: {query_text}
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, sources, context
        
    except Exception as e:
        return f"Lỗi: {str(e)}", [], ""

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần tìm hiểu thủ tục gì?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            answer, sources, debug_context = query_rag(prompt, SELECTED_MODEL)
            
            unique_sources = list(set(sources))
            if unique_sources:
                full_response = f"{answer}\n\n**📚 Nguồn tham khảo:**\n" + "\n".join(unique_sources)
            else:
                full_response = answer
            
            st.markdown(full_response)
            
            # Debug xem AI đọc được gì
            with st.expander("🕵️ Xem dữ liệu tìm được"):
                st.text(debug_context)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
