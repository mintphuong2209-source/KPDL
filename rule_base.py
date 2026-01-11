__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =============================================================================

import streamlit as st
import json
import os
import glob # Thư viện để tìm tất cả file
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Chatbot Tổng Hợp Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Pháp Luật & Cư Trú")

COLLECTION_NAME = "all_files_db_v1" # Tên kho dữ liệu chung

# ================= 2. CẤU HÌNH API & MODEL =================
st.sidebar.header("⚙️ Cấu hình")

api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# Tự động chọn model (Ưu tiên Flash)
try:
    models = [m.name for m in genai.list_models() if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods]
    default_idx = 0
    for i, m in enumerate(models):
        if "flash" in m: default_idx = i; break
    
    SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", models, index=default_idx)
except Exception as e:
    st.sidebar.error(f"Lỗi kết nối API: {e}")
    st.stop()

# ================= 3. HÀM LOAD TOÀN BỘ FILE JSON =================
@st.cache_resource(ttl="2h") 
def load_all_json_files():
    """
    Hàm này quét toàn bộ file .json trong thư mục và nạp vào ChromaDB.
    Tuyệt đối KHÔNG vẽ UI trong hàm này để tránh lỗi Cache.
    """
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    
    try:
        # 1. Khởi tạo ChromaDB
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # 2. Chỉ nạp nếu DB rỗng
        if collection.count() == 0:
            # --- TỰ ĐỘNG TÌM FILE ---
            json_files = glob.glob("*.json") # Lấy tất cả file có đuôi .json
            
            if not json_files:
                return "NO_FILES"

            all_data = []
            seen_content = set() # Dùng để lọc trùng lặp giữa các file
            
            # Đọc từng file và gộp lại
            for file_name in json_files:
                try:
                    with open(file_name, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                content = item.get("content_text", "").strip()
                                # Chỉ lấy nếu nội dung chưa từng xuất hiện và đủ dài
                                if content and content not in seen_content and len(content) > 10:
                                    seen_content.add(content)
                                    # Đánh dấu nguồn gốc file để dễ debug
                                    item["metadata"]["source_file"] = file_name 
                                    all_data.append(item)
                except:
                    pass # Bỏ qua file lỗi

            if not all_data:
                return "EMPTY_DATA"

            # 3. Chuẩn bị dữ liệu nạp
            ids = []
            documents = []
            metadatas = []
            
            for item in all_data:
                # Tạo ID
                if "id" in item:
                    ids.append(str(item["id"]))
                else:
                    ids.append(str(uuid.uuid4()))
                
                documents.append(item.get("content_text", ""))
                
                # Xử lý metadata
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                    "source_file": item.get("metadata", {}).get("source_file", "")
                })
                # Làm sạch metadata (xóa None)
                clean_meta = {k: (str(v) if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # 4. Nạp Batch (40 items/lần)
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

# --- GỌI HÀM NẠP VÀ HIỂN THỊ (BÊN NGOÀI CACHE) ---
with st.spinner("Đang quét và tổng hợp dữ liệu từ tất cả các file..."):
    collection = load_all_json_files()

# Xử lý các trạng thái
if collection == "NO_FILES":
    st.error("❌ Không tìm thấy file .json nào trên GitHub.")
    st.stop()
elif collection == "EMPTY_DATA":
    st.error("❌ Các file JSON đều rỗng hoặc lỗi định dạng.")
    st.stop()
elif collection is None:
    st.error("❌ Lỗi hệ thống khi khởi tạo ChromaDB.")
    st.stop()
else:
    # Hiển thị thành công
    count = collection.count()
    st.sidebar.success(f"📚 Tổng dữ liệu: **{count}** chunks")
    
    # Liệt kê các file tìm thấy (để bạn kiểm tra)
    with st.sidebar.expander("📂 Các file đã đọc"):
        found_files = glob.glob("*.json")
        for f in found_files:
            st.write(f"- `{f}`")

# ================= 4. LOGIC RAG =================
def query_rag(query_text, model_name, top_k=12): # Lấy 12 chunk để đủ thông tin
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
                src_file = meta.get('source_file', 'unknown')
                
                context_parts.append(f"--- (File: {src_file}) | {h} ---\n{doc}")
                sources.append(f"- [{h}] (Nguồn: {src_file})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hỗ trợ pháp lý. 
        Dựa vào các văn bản pháp luật được cung cấp dưới đây, hãy trả lời câu hỏi.
        
        YÊU CẦU:
        1. Trả lời chi tiết, có căn cứ.
        2. Nếu thông tin đến từ nhiều nguồn (nhiều file), hãy tổng hợp lại.
        3. Văn phong rõ ràng, dễ hiểu.
        
        NGỮ CẢNH DỮ LIỆU:
        {context}
        
        CÂU HỎI: {query_text}
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, list(set(sources)), context
        
    except Exception as e:
        return f"Lỗi: {str(e)}", [], ""

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi đã đọc hết các file dữ liệu của bạn. Bạn cần hỏi gì?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tổng hợp thông tin..."):
            answer, sources, debug_ctx = query_rag(prompt, SELECTED_MODEL)
            
            if sources:
                full_resp = f"{answer}\n\n**📚 Nguồn tham khảo:**\n" + "\n".join(sources)
            else:
                full_resp = answer
            
            st.markdown(full_resp)
            
            with st.expander("🕵️ Xem dữ liệu tìm được"):
                st.text(debug_ctx)
            
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
