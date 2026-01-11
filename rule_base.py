# ================= FIX LỖI SQLITE TRÊN STREAMLIT CLOUD =================
# Bắt buộc phải để 3 dòng này ở trên cùng, trước khi import chromadb
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
# Tên file dữ liệu bạn đã upload lên GitHub
JSON_FILE = "all_procedures_normalized.json" 
COLLECTION_NAME = "dichvucong_rag"

# Cấu hình Page
st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="centered")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")

# ================= XỬ LÝ API KEY =================
# Ưu tiên lấy từ Secrets, nếu không có thì hiện ô nhập
api_key = st.secrets.get("GEMINI_API_KEY") 

if not api_key:
    api_key = st.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.info("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# ================= HÀM LOAD DỮ LIỆU & VECTOR DB =================
@st.cache_resource
def initialize_vector_db():
    # Sử dụng model nhẹ hơn để tránh bị sập (Out of Memory) trên Cloud Free
    # Nếu muốn dùng BAAI/bge-m3 mà bị lỗi restart app, hãy đổi dòng dưới thành: "keepitreal/vietnamese-sbert"
    EMBEDDING_MODEL = "BAAI/bge-m3" 
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # Dùng Client ephermeral (chạy trên RAM)
    chroma_client = chromadb.Client()
    
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # Chỉ nạp dữ liệu nếu Collection đang rỗng
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                st.error(f"⚠️ Không tìm thấy file: {JSON_FILE}. Hãy upload file này lên GitHub cùng chỗ với app.py")
                return None
                
            with st.spinner("Đang nạp dữ liệu lần đầu (có thể mất 1-2 phút)..."):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Batch processing để nạp nhanh hơn
                ids = []
                documents = []
                metadatas = []
                
                for item in data:
                    ids.append(item["id"])
                    documents.append(item["content_text"])
                    
                    # Xử lý metadata an toàn
                    meta = item.get("metadata", {}).copy()
                    meta.update({
                        "url": item.get("url", ""),
                        "title": item.get("title", ""),
                        "hierarchy": item.get("hierarchy", ""),
                    })
                    # Xóa giá trị None để tránh lỗi Chroma
                    clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
                    metadatas.append(clean_meta)
                
                # Nạp theo lô 100 item/lần
                batch_size = 100
                total_batches = len(ids) // batch_size + 1
                progress_bar = st.progress(0)
                
                for i in range(0, len(ids), batch_size):
                    collection.add(
                        ids=ids[i:i+batch_size],
                        documents=documents[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size]
                    )
                    # Cập nhật thanh tiến trình
                    current_progress = min((i + batch_size) / len(ids), 1.0)
                    progress_bar.progress(current_progress)
                
                progress_bar.empty() # Xóa thanh tiến trình khi xong
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
collection = initialize_vector_db()

if not collection:
    st.stop()

# ================= LOGIC RAG & CHAT =================
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
        Bạn là trợ lý ảo hành chính công. Hãy trả lời dựa trên thông tin sau:
        
        NGỮ CẢNH:
        {context}
        
        CÂU HỎI: {query_text}
        
        YÊU CẦU: Trả lời ngắn gọn, chính xác bằng tiếng Việt. Nếu không có thông tin, hãy nói không biết.
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text, sources
    except Exception as e:
        return f"Xin lỗi, hệ thống đang bận. Lỗi: {str(e)}", []

# ================= GIAO DIỆN CHAT =================
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
            
            # Xử lý nguồn trùng lặp
            unique_sources = list(set(sources))
            
            if unique_sources:
                full_response = f"{answer}\n\n**Nguồn tham khảo:**\n" + "\n".join(unique_sources)
            else:
                full_response = answer
                
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
