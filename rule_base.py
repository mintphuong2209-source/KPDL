# ================= 1. FIX LỖI SQLITE (BẮT BUỘC TRÊN STREAMLIT CLOUD) =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ======================================================================================

import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG (ĐÁP ỨNG CÂU 5) =================
st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", page_icon="🇻🇳", layout="wide")
st.title("🤖 Trợ Lý Ảo Tư Vấn Pháp Luật Cư Trú")
st.markdown("---")

# Cấu hình đường dẫn file data
JSON_FILE = "all_chunks_normalized.json"
COLLECTION_NAME = "dichvucong_rag_v1"

# ================= 2. CẤU HÌNH API (ĐÁP ỨNG CÂU 4) =================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    # Lấy API Key từ Secrets hoặc nhập tay
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Nhập Google AI API Key:", type="password")
    
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()
    
    genai.configure(api_key=api_key)
    
    # Cho phép chỉnh số lượng chunk retrieval (Advanced)
    top_k = st.slider("Số lượng nguồn tham khảo (Top-K):", 1, 10, 5)

# ================= 3. HÀM NẠP DỮ LIỆU (OPTIMIZED) =================
@st.cache_resource(show_spinner=False)
def load_vector_db():
    try:
        # Sử dụng model embedding tối ưu cho tiếng Việt (nhẹ hơn bge-m3 để chạy cloud)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="keepitreal/vietnamese-sbert"
        )
        
        # Tạo Client (Ephemeral - chạy trên RAM để tốc độ cao nhất)
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        # Kiểm tra nếu DB rỗng thì nạp từ file JSON
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                return None, f"Không tìm thấy file {JSON_FILE}!"

            with open(JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            ids = []
            documents = []
            metadatas = []

            # Thanh tiến trình
            progress_bar = st.progress(0, text="Đang nạp dữ liệu tri thức...")
            total = len(data)

            for idx, item in enumerate(data):
                # Lọc bỏ chunk rác
                content = item.get("content_text", "").strip()
                if len(content) < 10: continue

                ids.append(str(item.get("id", idx)))
                documents.append(content)
                
                # Xử lý Metadata kỹ càng để hiển thị nguồn (Câu 5)
                meta_source = item.get("metadata", {})
                clean_meta = {
                    "url": str(item.get("url", "#")),
                    "hierarchy": str(item.get("hierarchy", "Thông tin chung")),
                    "title": str(item.get("title", "")),
                    "source_domain": str(meta_source.get("source_domain", "DVCQG"))
                }
                metadatas.append(clean_meta)
                
                # Cập nhật thanh tiến trình mỗi 10%
                if idx % (total // 10 + 1) == 0:
                    progress_bar.progress(idx / total)

            # Nạp vào Chroma (Batch size 100)
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
            
            progress_bar.empty() # Xóa thanh tiến trình khi xong
            return collection, f"Đã nạp mới {len(ids)} chunks."
        
        return collection, f"Sẵn sàng ({collection.count()} chunks)."

    except Exception as e:
        return None, str(e)

# --- KHỞI ĐỘNG DATABASE ---
with st.spinner("🚀 Đang khởi động hệ thống..."):
    collection, status_msg = load_vector_db()

if not collection:
    st.error(f"❌ Lỗi hệ thống: {status_msg}")
    st.stop()
else:
    st.sidebar.success(f"📦 Trạng thái: {status_msg}")

# ================= 4. LOGIC RAG & GEMINI (CÂU 4) =================
def query_gemini(question, k_neighbors):
    # 1. Retrieval
    results = collection.query(
        query_texts=[question],
        n_results=k_neighbors
    )
    
    context_parts = []
    sources_debug = [] # Để hiển thị metadata chi tiết
    
    if results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            # Format: [Nguồn] Nội dung
            source_info = f"[{meta['hierarchy']}]"
            context_parts.append(f"{source_info}\n{doc}")
            
            # Lưu lại để hiển thị UI
            sources_debug.append({
                "title": meta['hierarchy'],
                "url": meta['url'],
                "content": doc,
                "domain": meta['source_domain']
            })
            
    context_text = "\n\n".join(context_parts)
    
    # 2. Generation (Prompt Engineering)
    prompt = f"""
    Bạn là trợ lý ảo pháp luật chuyên về Cư trú.
    Dựa vào các đoạn văn bản pháp luật được cung cấp dưới đây, hãy trả lời câu hỏi của công dân.
    
    YÊU CẦU:
    1. Trả lời chính xác, ngắn gọn, dễ hiểu.
    2. Dẫn chứng thông tin lấy từ mục nào (dựa vào phần [...] trong context).
    3. Nếu không có thông tin trong ngữ cảnh, hãy nói "Tôi chưa tìm thấy thông tin trong dữ liệu hiện có".
    
    NGỮ CẢNH (CONTEXT):
    {context_text}
    
    CÂU HỎI: {question}
    """
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash") # Thử gọi bản mới nhất
        response = model.generate_content(prompt)
        return response.text, sources_debug
    except:
        # Fallback về bản ổn định nếu bản 2.5 lỗi
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text, sources_debug
        except Exception as e:
            return f"⚠️ Lỗi kết nối AI: {str(e)}", []

# ================= 5. GIAO DIỆN CHAT (CÂU 5) =================
# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì về thủ tục đăng ký thường trú, tạm trú, hay tách hộ?"}]

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Nếu có thông tin debug (nguồn) từ tin nhắn cũ thì hiển thị lại
        if "sources" in msg:
            with st.expander("🔍 Xem nguồn & Metadata (Evidence)"):
                for s in msg["sources"]:
                    st.markdown(f"**📑 {s['title']}**")
                    st.caption(f"Nguồn: {s['domain']} | [Xem chi tiết]({s['url']})")
                    st.text(f"{s['content'][:150]}...") # Trích dẫn 1 đoạn ngắn
                    st.divider()

# Xử lý input mới
if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu văn bản pháp luật..."):
            answer, sources = query_gemini(prompt, top_k)
            
            st.markdown(answer)
            
            # Hiển thị nguồn chi tiết (Ăn điểm câu hiển thị metadata/chunk)
            if sources:
                with st.expander("🔍 Xem nguồn & Metadata (Evidence)"):
                    st.info("Các đoạn văn bản được AI sử dụng để trả lời:")
                    for s in sources:
                        st.markdown(f"**📑 {s['title']}**")
                        st.caption(f"Nguồn: {s['domain']} | [Xem chi tiết]({s['url']})")
                        st.text(f"Nội dung chunk: {s['content'][:200]}...") # Show 200 ký tự đầu
                        st.divider()
            
            # Lưu vào lịch sử kèm nguồn
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
