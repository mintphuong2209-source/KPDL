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

# Tên file dữ liệu chuẩn của bạn
JSON_FILE = "all_chunks_normalized.json" 
COLLECTION_NAME = "dichvucong_rag_final"

# ================= 2. CẤU HÌNH API & MODEL =================
st.sidebar.header("⚙️ Cấu hình")

api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# Tự động tìm model (Ưu tiên Flash)
try:
    available_models = [m.name for m in genai.list_models() if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods]
    default_idx = 0
    for i, m in enumerate(available_models):
        if "flash" in m: default_idx = i; break
    
    if available_models:
        SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", available_models, index=default_idx)
    else:
        st.sidebar.error("❌ Không tìm thấy model Gemini nào.")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Lỗi API: {e}")
    st.stop()

# ================= 3. HÀM LOAD DỮ LIỆU (KHÔNG CÓ UI BÊN TRONG) =================
@st.cache_resource(ttl="2h") 
def get_vector_collection():
    """
    Hàm này chỉ thực hiện logic nạp dữ liệu, tuyệt đối KHÔNG vẽ UI (st.write, st.progress...)
    để tránh lỗi CacheReplayClosureError.
    """
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    
    try:
        # 1. Khởi tạo Client
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # 2. Kiểm tra dữ liệu
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                return None # Trả về None để xử lý lỗi ở ngoài
            
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not data or not isinstance(data, list):
                return None

            # 3. Chuẩn bị Batch
            ids = [str(uuid.uuid4()) if "id" not in item else str(item["id"]) for item in data]
            documents = [item.get("content_text", "") for item in data]
            metadatas = []
            
            for item in data:
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                })
                # Xóa giá trị None để tránh lỗi Chroma
                clean_meta = {k: (str(v) if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # 4. Nạp dữ liệu (Không dùng progress bar ở đây)
            batch_size = 40
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
                
        return collection
        
    except Exception as e:
        print(f"Lỗi nạp DB: {e}")
        return None

# --- GỌI HÀM NẠP DỮ LIỆU ---
# Đặt st.spinner ở ngoài hàm cache
with st.spinner("Đang khởi tạo bộ nhớ tri thức (Lần đầu sẽ mất khoảng 1-2 phút)..."):
    collection = get_vector_collection()

# Xử lý trường hợp lỗi
if collection is None:
    st.error(f"❌ Không thể nạp dữ liệu. Vui lòng kiểm tra file `{JSON_FILE}` trên GitHub.")
    st.stop()

# Hiển thị thống kê
st.sidebar.success(f"📦 Dữ liệu đã nạp: **{collection.count()}** chunks")

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
                context_parts.append(f"--- MỤC: {h} ---\n{doc}")
                sources.append(f"- [{h}]({u})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hành chính công chuyên về Luật Cư trú.
        Hãy trả lời câu hỏi của công dân dựa trên thông tin được cung cấp dưới đây.
        
        NGUYÊN TẮC:
        1. Trả lời chính xác, ngắn gọn, dễ hiểu.
        2. Nếu là quy trình, hãy liệt kê các bước (Bước 1, Bước 2...).
        3. Nếu hồ sơ yêu cầu giấy tờ, hãy liệt kê bằng gạch đầu dòng.
        4. Tuyệt đối không bịa đặt thông tin nếu không có trong ngữ cảnh.
        
        NGỮ CẢNH THÔNG TIN:
        {context}
        
        CÂU HỎI: {query_text}
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, list(set(sources)), context
        
    except Exception as e:
        return f"Hệ thống đang bận: {str(e)}", [], ""

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Mình là trợ lý ảo hỗ trợ thủ tục cư trú (Thường trú, Tạm trú, Tách hộ...). Bạn cần giúp gì không?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu luật..."):
            answer, sources, debug_ctx = query_rag(prompt, SELECTED_MODEL)
            
            if sources:
                full_resp = f"{answer}\n\n**📚 Nguồn tham khảo:**\n" + "\n".join(sources)
            else:
                full_resp = answer
            
            st.markdown(full_resp)
            
            with st.expander("🕵️ Xem dữ liệu hệ thống tìm được"):
                st.text(debug_ctx)
            
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
