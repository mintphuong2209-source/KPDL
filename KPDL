import streamlit as st
import json
import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH =================
# Tên file dữ liệu bạn đã upload lên GitHub (nằm cùng thư mục với app.py)
JSON_FILE = "all_procedures_normalized.json" 
COLLECTION_NAME = "dichvucong_rag"

# Cấu hình Page
st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="centered")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")

# ================= XỬ LÝ API KEY =================
# Cách 1: Lấy từ st.secrets (Khuyên dùng khi deploy lên Streamlit Cloud)
# Cách 2: Nhập trực tiếp (Chỉ dùng test nhanh, không bảo mật)
api_key = st.secrets.get("GEMINI_API_KEY") 

if not api_key:
    api_key = st.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.info("Vui lòng nhập API Key để tiếp tục.")
        st.stop()

genai.configure(api_key=api_key)

# ================= HÀM LOAD DỮ LIỆU & VECTOR DB =================
# Dùng @st.cache_resource để chỉ chạy 1 lần duy nhất khi khởi động app
@st.cache_resource
def initialize_vector_db():
    # 1. Khởi tạo Embedding Model (Dùng model nhẹ hơn chút để chạy mượt trên Cloud Free)
    # Nếu muốn chính xác cao hơn nhưng chậm hơn, đổi lại thành "BAAI/bge-m3"
    EMBEDDING_MODEL = "BAAI/bge-m3" 
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    # 2. Tạo Client ChromaDB (Chạy trong memory để tránh lỗi path trên Cloud)
    chroma_client = chromadb.Client()
    
    try:
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # 3. Kiểm tra xem đã có dữ liệu chưa, nếu chưa thì nạp từ JSON
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                st.error(f"⚠️ Không tìm thấy file dữ liệu: {JSON_FILE}. Vui lòng upload file này lên GitHub.")
                return None
                
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Chuẩn bị dữ liệu để nạp (Batch processing để tránh quá tải RAM)
            ids = []
            documents = []
            metadatas = []
            
            # Giới hạn số lượng nếu file quá lớn (Ví dụ test với 500 dòng đầu tiên)
            # data = data[:500] 
            
            for item in data:
                ids.append(item["id"])
                documents.append(item["content_text"])
                
                # Xử lý metadata
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                })
                # Đảm bảo metadata không chứa None value (ChromaDB không chịu None)
                clean_meta = {k: (v if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # Nạp vào DB
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo Vector DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
with st.spinner("Đang khởi tạo cơ sở dữ liệu tri thức (Lần đầu sẽ hơi lâu)..."):
    collection = initialize_vector_db()

if not collection:
    st.stop()

# ================= LOGIC RAG & CHAT =================
def query_rag(query_text, top_k=3):
    # 1. Tìm kiếm trong Vector DB
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    
    # 2. Tạo Context
    context_parts = []
    sources = []
    
    if results["documents"]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hierarchy = meta.get('hierarchy', meta.get('title', 'Mục'))
            url = meta.get('url', '#')
            context_parts.append(f"[{hierarchy}]\n{doc}")
            sources.append(f"- [{hierarchy}]({url})")
            
    context = "\n\n".join(context_parts)
    
    # 3. Tạo Prompt
    prompt = f"""
    Bạn là trợ lý ảo hỗ trợ pháp lý về cư trú Việt Nam.
    Dựa vào thông tin sau đây để trả lời câu hỏi. 
    Nếu không có thông tin trong ngữ cảnh, hãy nói "Tôi chưa tìm thấy thông tin trong văn bản pháp luật hiện có".
    
    NGỮ CẢNH:
    {context}
    
    CÂU HỎI: {query_text}
    
    TRẢ LỜI (Ngắn gọn, format đẹp):
    """
    
    # 4. Gọi Gemini
    model = genai.GenerativeModel('gemini-1.5-flash') # Dùng bản Flash cho nhanh và rẻ
    response = model.generate_content(prompt)
    return response.text, sources

# ================= GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì về thủ tục thường trú, tạm trú?"}]

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Xử lý khi người dùng nhập
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiện câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu luật..."):
            try:
                answer, sources = query_rag(prompt)
                
                # Format câu trả lời kèm nguồn
                full_response = f"{answer}\n\n**Nguồn tham khảo:**\n" + "\n".join(set(sources))
                
                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")
