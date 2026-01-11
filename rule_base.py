# ================= FIX LỖI SQLITE TRÊN STREAMLIT CLOUD =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =======================================================================

import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH =================
JSON_FILE = "all_procedures_normalized.json" 
COLLECTION_NAME = "dichvucong_rag"

st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="centered")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")

# ================= XỬ LÝ API KEY =================
api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.info("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# ================= HÀM LOAD DỮ LIỆU TỐI ƯU RAM =================
@st.cache_resource
def initialize_vector_db():
    # 👉 DÙNG MODEL NHẸ ĐỂ KHÔNG BỊ SẬP APP
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
        
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                st.error(f"⚠️ Không tìm thấy file: {JSON_FILE}")
                return None
                
            with st.spinner("Đang nạp dữ liệu (Chế độ tiết kiệm RAM)..."):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Giới hạn dữ liệu nếu file quá lớn (ví dụ chỉ lấy 1000 dòng đầu để test)
                # data = data[:1000] 

                ids = [item["id"] for item in data]
                documents = [item["content_text"] for item in data]
                metadatas = []
                
                for item in data:
                    meta = item.get("metadata", {}).copy()
                    meta.update({
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "hierarchy": item.get("hierarchy", ""),
                    })
                    # Xóa None value
                    clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
                    metadatas.append(clean_meta)
                
                # 👉 GIẢM BATCH SIZE XUỐNG 40 ĐỂ KHÔNG TRÀN RAM
                batch_size = 40
                progress_bar = st.progress(0)
                
                for i in range(0, len(ids), batch_size):
                    collection.add(
                        ids=ids[i:i+batch_size],
                        documents=documents[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size]
                    )
                    progress_bar.progress(min((i + batch_size) / len(ids), 1.0))
                
                progress_bar.empty()
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
collection = initialize_vector_db()

if not collection:
    st.stop()

# ================= LOGIC CHAT =================
def query_rag(query_text, top_k=3):
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
                context_parts.append(f"[{hierarchy}]\n{doc}")
                sources.append(f"- [{hierarchy}]({url})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hành chính công. Hãy trả lời câu hỏi dựa trên thông tin sau:
        
        THÔNG TIN:
        {context}
        
        CÂU HỎI: {query_text}
        
        YÊU CẦU: Trả lời ngắn gọn, chính xác bằng tiếng Việt.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as e:
        return f"Lỗi hệ thống: {str(e)}", []

# ================= GIAO DIỆN =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần tìm hiểu về thủ tục cư trú nào?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang suy nghĩ..."):
            answer, sources = query_rag(prompt)
            full_response = f"{answer}\n\n**Nguồn:**\n" + "\n".join(list(set(sources)))
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
