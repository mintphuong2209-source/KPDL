# --- BẮT BUỘC: FIX LỖI SQLITE TRÊN STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------------------

import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Chatbot Thủ Tục Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú (RAG)")

# ================= CẤU HÌNH API & DATA =================
# Lấy API Key từ Secrets của Streamlit
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("Chưa cấu hình GEMINI_API_KEY trong Secrets!")
    st.stop()

# Đường dẫn file data (bạn phải upload file này vào thư mục data trên github)
JSON_FILE = "data/all_procedures_normalized.json"
COLLECTION_NAME = "dichvucong_rag_collection"

# ================= HÀM LOAD DỮ LIỆU (CACHED) =================
@st.cache_resource
def load_vector_db():
    """
    Khởi tạo ChromaDB và nạp dữ liệu từ file JSON.
    Dùng cache để không phải nạp lại mỗi lần reload trang.
    """
    # Sử dụng model nhẹ hơn bge-m3 một chút để chạy mượt trên Cloud Free
    # Hoặc bạn có thể giữ nguyên "BAAI/bge-m3" nếu muốn
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert" 
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Sử dụng EphemeralClient (chạy trên RAM) cho môi trường Cloud
    chroma_client = chromadb.Client()
    
    # Tạo hoặc lấy collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    # Kiểm tra nếu collection rỗng thì mới nạp
    if collection.count() == 0:
        if not os.path.exists(JSON_FILE):
            st.error(f"Không tìm thấy file dữ liệu tại {JSON_FILE}")
            return None

        with open(JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        ids = []
        documents = []
        metadatas = []

        # Chuẩn bị dữ liệu (Batch processing để tránh quá tải RAM)
        batch_size = 100
        total_chunks = len(data)
        
        progress_text = "Đang nạp dữ liệu vào bộ nhớ..."
        my_bar = st.progress(0, text=progress_text)

        for idx, item in enumerate(data):
            ids.append(str(item.get("id", idx))) # Đảm bảo ID là string
            documents.append(item["content_text"])
            
            # Xử lý metadata (Chroma không nhận None value, phải chuyển thành string rỗng)
            meta = item.get("metadata", {}).copy()
            meta.update({
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "hierarchy": item.get("hierarchy", ""),
                "chunk_type": item.get("chunk_type", ""),
            })
            # Clean metadata values
            clean_meta = {k: str(v) if v is not None else "" for k, v in meta.items()}
            metadatas.append(clean_meta)

        # Add to Chroma theo batch
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            my_bar.progress(min(i / total_chunks, 1.0), text=f"Đã nạp {i}/{total_chunks} chunks")
        
        my_bar.empty()
        
    return collection

# ================= HÀM TRUY VẤN (RAG) =================
def query_gemini(question, collection, model_name="gemini-2.5-flash"):
    # 1. Truy vấn Vector DB
    results = collection.query(
        query_texts=[question],
        n_results=5, # Lấy 5 đoạn liên quan nhất
        include=["documents", "metadatas"]
    )

    # 2. Xây dựng Context
    context_parts = []
    sources = []
    
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hierarchy = meta.get('hierarchy', 'Thông tin')
            url = meta.get('url', '#')
            context_parts.append(f"[{hierarchy}]\n{doc}")
            sources.append(f"- [{hierarchy}]({url})")

    context = "\n\n".join(context_parts)

    if not context:
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu của mình.", []

    # 3. Tạo Prompt
    prompt = f"""
    Bạn là trợ lý tư vấn thủ tục hành chính công của Việt Nam (lĩnh vực Cư trú).
    
    NGUYÊN TẮC:
    - Chỉ sử dụng thông tin có trong CONTEXT bên dưới.
    - Không bịa đặt thông tin. Nếu không có trong context, hãy nói không biết.
    - Trả lời ngắn gọn, rõ ràng, đánh số bước nếu cần.
    
    CONTEXT:
    {context}
    
    CÂU HỎI: {question}
    """

    # 4. Gọi Gemini
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, list(set(sources)) # Trả về câu trả lời và nguồn (unique)
    except Exception as e:
        return f"Lỗi kết nối Gemini: {str(e)}", []

# ================= GIAO DIỆN CHÍNH =================
# Load Database
collection = load_vector_db()

if collection:
    st.sidebar.success(f"Dữ liệu đã sẵn sàng: {collection.count()} chunks")
else:
    st.stop()

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì về thủ tục Thường trú, Tạm trú, Tách hộ...?"}]

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý input người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiển thị câu hỏi user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu luật..."):
            response_text, sources = query_gemini(prompt, collection)
            
            # Format câu trả lời kèm nguồn
            final_content = response_text
            if sources:
                final_content += "\n\n**Nguồn tham khảo:**\n" + "\n".join(sources)
            
            st.markdown(final_content)
            st.session_state.messages.append({"role": "assistant", "content": final_content})
