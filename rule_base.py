# ================= FIX LỖI SQLITE TRÊN STREAMLIT CLOUD =================
# BẮT BUỘC: Phải để 3 dòng này ở trên cùng file
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

st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")

# ================= 1. XỬ LÝ API KEY & CHỌN MODEL (AUTO) =================
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
            # Ưu tiên các model Pro hoặc Flash
            if 'gemini' in m.name:
                available_models.append(m.name)
    
    if available_models:
        # Cho phép người dùng chọn model nếu tìm thấy nhiều
        SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", available_models, index=0)
    else:
        st.sidebar.error("❌ Không tìm thấy model Gemini nào cho Key này.")
        st.stop()
        
except Exception as e:
    st.sidebar.error(f"Lỗi kết nối API: {e}")
    st.stop()

# ================= 2. HÀM LOAD DỮ LIỆU TỐI ƯU RAM =================
@st.cache_resource
def initialize_vector_db():
    # 👉 DÙNG MODEL NHẸ NHẤT ĐỂ KHÔNG BỊ OUT OF MEMORY
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    
    try:
        # Load embedding model
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Tạo Client ChromaDB (Chạy trên RAM - Ephemeral)
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # Nếu chưa có dữ liệu thì nạp mới
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                st.error(f"⚠️ Không tìm thấy file: {JSON_FILE}. Hãy upload file này lên GitHub.")
                return None
                
            with st.spinner("Đang nạp dữ liệu (Chế độ tiết kiệm RAM)..."):
                with open(JSON_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Tách dữ liệu ra các list
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
                    # Xóa giá trị None (ChromaDB không chịu None)
                    clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
                    metadatas.append(clean_meta)
                
                # 👉 Nạp Batch nhỏ (40) để tránh tràn RAM
                batch_size = 40
                progress_bar = st.sidebar.progress(0)
                
                for i in range(0, len(ids), batch_size):
                    collection.add(
                        ids=ids[i:i+batch_size],
                        documents=documents[i:i+batch_size],
                        metadatas=metadatas[i:i+batch_size]
                    )
                    # Cập nhật thanh tiến trình
                    progress = min((i + batch_size) / len(ids), 1.0)
                    progress_bar.progress(progress)
                
                progress_bar.empty()
                st.toast(f"Đã nạp xong {len(ids)} chunks!", icon="✅")
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
collection = initialize_vector_db()

if not collection:
    st.stop()
    
# HIỆN SỐ LƯỢNG CHUNK LÊN SIDEBAR
st.sidebar.success(f"📦 Dữ liệu đã nạp: **{collection.count()}** chunks")

# ================= 3. LOGIC HỎI ĐÁP (RAG) =================
def query_rag(query_text, model_name, top_k=12): # 👉 Tăng top_k lên 12 để lấy nhiều thông tin hơn
    try:
        # Tìm kiếm vector
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
                
                # Tạo context
                context_parts.append(f"--- NỘI DUNG TỪ MỤC: {hierarchy} ---\n{doc}")
                
                # Lưu nguồn
                sources.append(f"- [{hierarchy}]({url})")
                
        context = "\n\n".join(context_parts)
        
        # Prompt chi tiết
        prompt = f"""
        Bạn là trợ lý ảo hỗ trợ pháp lý về thủ tục hành chính cư trú tại Việt Nam.
        Nhiệm vụ: Trả lời câu hỏi của người dùng CHỈ dựa trên thông tin được cung cấp dưới đây.
        
        YÊU CẦU:
        1. Trả lời chi tiết, từng bước nếu là quy trình.
        2. Nếu thông tin có trong ngữ cảnh, hãy trích dẫn.
        3. Nếu KHÔNG tìm thấy thông tin trong ngữ cảnh, hãy nói: "Xin lỗi, tôi chưa tìm thấy thông tin cụ thể trong dữ liệu hiện có."
        4. Không tự bịa ra thông tin pháp luật.

        NGỮ CẢNH THÔNG TIN (Dữ liệu tìm được):
        {context}
        
        CÂU HỎI CỦA NGƯỜI DÙNG: {query_text}
        
        TRẢ LỜI:
        """
        
        # Gọi Gemini với model đã chọn tự động
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, sources, context
        
    except Exception as e:
        return f"Lỗi xử lý: {str(e)}", [], ""

# ================= 4. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần tìm hiểu thủ tục gì (ví dụ: Đăng ký thường trú, Tách hộ...)? "}]

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý input
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bot response
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu dữ liệu pháp luật..."):
            answer, sources, debug_context = query_rag(prompt, SELECTED_MODEL)
            
            # Xử lý nguồn tham khảo (xóa trùng lặp)
            unique_sources = list(set(sources))
            
            if unique_sources:
                full_response = f"{answer}\n\n---\n**📚 Nguồn tham khảo:**\n" + "\n".join(unique_sources)
            else:
                full_response = answer
            
            st.markdown(full_response)
            
            # Debug: Cho phép xem những gì AI đã đọc được (để kiểm tra xem nó có đọc đúng chunk không)
            with st.expander("🕵️ [Debug] Xem dữ liệu AI tìm thấy"):
                st.text(debug_context)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
