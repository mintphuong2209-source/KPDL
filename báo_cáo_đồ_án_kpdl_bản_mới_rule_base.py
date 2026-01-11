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

# ... (Phần import giữ nguyên như cũ, nhớ 3 dòng fix sqlite ở đầu) ...

# ================= 3. HÀM LOAD DỮ LIỆU (CÓ BÁO LỖI CHI TIẾT) =================
@st.cache_resource(ttl="2h") 
def load_all_json_files():
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
        
        # 2. Tìm file JSON
        json_files = glob.glob("*.json")
        if not json_files:
            return "NO_FILES_FOUND" # Mã lỗi riêng

        # 3. Đọc dữ liệu
        if collection.count() == 0:
            ids, documents, metadatas = [], [], []
            
            for file_name in json_files:
                with open(file_name, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        # Lọc dữ liệu rỗng
                        if not item.get("content_text"): continue
                        
                        ids.append(str(uuid.uuid4()))
                        documents.append(item.get("content_text"))
                        
                        # Xử lý metadata
                        meta = item.get("metadata", {}).copy()
                        meta["source_file"] = file_name
                        # Xóa giá trị None để tránh lỗi
                        clean_meta = {k: str(v) for k, v in meta.items() if v is not None}
                        metadatas.append(clean_meta)
            
            if not documents:
                return "EMPTY_DATA"

            # 4. Nạp Batch
            batch_size = 40
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
                
        return collection
        
    except Exception as e:
        # TRẢ VỀ CHI TIẾT LỖI ĐỂ DEBUG
        return f"ERROR_DETAIL: {str(e)}"

# --- GỌI HÀM (SỬA LẠI ĐỂ BẮT LỖI) ---
with st.spinner("Đang khởi động hệ thống..."):
    collection = load_all_json_files()

if isinstance(collection, str): # Nếu trả về chuỗi nghĩa là có lỗi
    if "ERROR_DETAIL" in collection:
        st.error(f"❌ LỖI HỆ THỐNG CHI TIẾT: {collection}")
        st.info("👉 Hãy chụp ảnh lỗi này gửi cho tôi để được hỗ trợ!")
    elif collection == "NO_FILES_FOUND":
        st.warning("⚠️ Không tìm thấy file .json nào trên GitHub. Bạn đã upload file chưa?")
    st.stop()

# Nếu thành công
st.sidebar.success(f"✅ Đã nạp: **{collection.count()}** chunks")

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
