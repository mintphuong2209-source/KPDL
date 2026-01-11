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
st.set_page_config(page_title="Hỏi Đáp Dịch Vụ Công", layout="wide")
st.title("🤖 Chatbot Tư Vấn Dịch Vụ Công & Cư Trú")

# 👉 TÊN FILE DỮ LIỆU MỚI CỦA BẠN
JSON_FILE = "all_chunks_normalized.json" 
COLLECTION_NAME = "dichvucong_db_v1" # Tên kho dữ liệu

# ================= 2. CẤU HÌNH API & MODEL =================
st.sidebar.header("⚙️ Cấu hình")

api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# Tự động chọn model tốt nhất (Ưu tiên Flash)
try:
    models = [m.name for m in genai.list_models() if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods]
    default_idx = 0
    for i, m in enumerate(models):
        if "flash" in m: default_idx = i; break
    
    SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", models, index=default_idx)
except Exception as e:
    st.sidebar.error(f"Lỗi kết nối API: {e}")
    st.stop()

# ================= 3. HÀM LOAD DỮ LIỆU (CORE LOGIC) =================
@st.cache_resource(ttl="2h") 
def get_vector_collection():
    """
    Hàm này chỉ nạp dữ liệu, KHÔNG được chứa lệnh vẽ giao diện (st.write, st.spinner...)
    để tránh lỗi CacheReplayClosureError.
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
        
        # 2. Kiểm tra nếu DB rỗng thì nạp từ file JSON
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                return None # Báo lỗi file không tồn tại
            
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not data or not isinstance(data, list):
                return "EMPTY_DATA" # Báo lỗi file rỗng

            # 3. Chuẩn bị dữ liệu
            ids = []
            documents = []
            metadatas = []
            
            for item in data:
                # Lấy nội dung text
                content = item.get("content_text", "")
                if not content: continue
                
                # Tạo ID duy nhất
                if "id" in item:
                    ids.append(str(item["id"]))
                else:
                    ids.append(str(uuid.uuid4()))
                
                documents.append(content)
                
                # Xử lý metadata (Chroma không nhận None)
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                })
                clean_meta = {k: (str(v) if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # 4. Nạp theo lô (Batch) để tiết kiệm RAM
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
        return "ERROR"

# --- GỌI HÀM NẠP VÀ HIỂN THỊ TRẠNG THÁI ---
with st.status("Đang khởi động hệ thống tri thức...", expanded=True) as status:
    st.write("🔄 Đang kết nối cơ sở dữ liệu...")
    collection = get_vector_collection()
    
    if collection is None:
        status.update(label="Lỗi khởi động!", state="error")
        st.error(f"❌ Không tìm thấy file `{JSON_FILE}`. Hãy upload file này lên GitHub.")
        st.stop()
    elif collection == "EMPTY_DATA":
        status.update(label="Lỗi dữ liệu!", state="error")
        st.error("❌ File JSON bị rỗng hoặc sai định dạng.")
        st.stop()
    elif collection == "ERROR":
        status.update(label="Lỗi hệ thống!", state="error")
        st.error("❌ Lỗi khi khởi tạo ChromaDB.")
        st.stop()
    else:
        count = collection.count()
        status.update(label="Sẵn sàng!", state="complete", expanded=False)
        st.sidebar.success(f"📦 Dữ liệu đã nạp: **{count}** chunks")

# ================= 4. LOGIC TÌM KIẾM & TRẢ LỜI =================
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
                source_name = meta.get('source', 'Nguồn')
                
                # Tạo ngữ cảnh cho AI đọc
                context_parts.append(f"--- NGUỒN: {source_name} | MỤC: {h} ---\n{doc}")
                sources.append(f"- [{h}]({u})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hành chính công chuyên nghiệp.
        Nhiệm vụ: Trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
        
        YÊU CẦU:
        1. Trả lời chi tiết, chính xác, không bịa đặt.
        2. Nếu là quy trình, hãy trình bày từng bước (Bước 1, Bước 2...).
        3. Văn phong lịch sự, dễ hiểu.
        
        DỮ LIỆU THAM KHẢO:
        {context}
        
        CÂU HỎI CỦA CÔNG DÂN: {query_text}
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, list(set(sources)), context
        
    except Exception as e:
        return f"Hệ thống đang bận: {str(e)}", [], ""

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì về thủ tục hành chính (Thường trú, Tạm trú, CCCD...)?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu quy định pháp luật..."):
            answer, sources, debug_ctx = query_rag(prompt, SELECTED_MODEL)
            
            if sources:
                full_resp = f"{answer}\n\n**📚 Nguồn tham khảo:**\n" + "\n".join(sources)
            else:
                full_resp = answer
            
            st.markdown(full_resp)
            
            # Debug: Xem AI đã đọc được gì (giúp bạn kiểm tra dữ liệu)
            with st.expander("🕵️ Dữ liệu trích xuất"):
                st.text(debug_ctx)
            
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
