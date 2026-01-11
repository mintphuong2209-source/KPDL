# ================= 1. FIX LỖI SQLITE TRÊN CLOUD (BẮT BUỘC) =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ===========================================================================

import streamlit as st
import json
import os
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Hỏi Đáp Thủ Tục Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú (Dữ liệu BCA)")

# Tên file dữ liệu (phải khớp với tên file bạn upload lên GitHub)
JSON_FILE = "all_chunks_normalized.json"
COLLECTION_NAME = "dichvucong_data_final"

# ================= 2. CẤU HÌNH API =================
st.sidebar.header("⚙️ Cấu hình")

# Lấy API Key từ Secrets (Ưu tiên) hoặc nhập tay
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# ================= 3. HÀM NẠP DỮ LIỆU (CORE LOGIC) =================
# Hàm này chỉ chạy 1 lần duy nhất khi khởi động app
@st.cache_resource(show_spinner=False)
def initialize_database():
    """
    Hàm khởi tạo Vector DB và nạp dữ liệu từ file JSON.
    Tuyệt đối KHÔNG dùng st.write, st.spinner ở trong hàm này để tránh lỗi Cache.
    """
    try:
        # 1. Cấu hình ChromaDB với model nhúng tiếng Việt
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="keepitreal/vietnamese-sbert"
        )
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )

        # 2. Kiểm tra nếu DB chưa có dữ liệu thì mới nạp
        if collection.count() == 0:
            if not os.path.exists(JSON_FILE):
                return None, f"FILE_NOT_FOUND: Không tìm thấy file {JSON_FILE}"

            with open(JSON_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data or not isinstance(data, list):
                return None, "INVALID_DATA: File JSON lỗi hoặc rỗng"

            # 3. Chuẩn bị dữ liệu để nạp
            ids = []
            documents = []
            metadatas = []

            for idx, item in enumerate(data):
                # Lấy nội dung
                content = item.get("content_text", "").strip()
                if not content: continue

                # Tạo ID (dùng ID trong file hoặc tạo mới)
                doc_id = str(item.get("id", uuid.uuid4().hex))
                
                ids.append(doc_id)
                documents.append(content)

                # Xử lý Metadata (Chroma không nhận giá trị None)
                raw_meta = item.get("metadata", {})
                clean_meta = {
                    "source_url": str(item.get("url", "")),
                    "title": str(item.get("title", "")),
                    "hierarchy": str(item.get("hierarchy", "")),
                    "source": str(raw_meta.get("source_domain", "BCA"))
                }
                metadatas.append(clean_meta)

            # 4. Nạp vào ChromaDB theo lô (Batch)
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i : i+batch_size],
                    documents=documents[i : i+batch_size],
                    metadatas=metadatas[i : i+batch_size]
                )
            
            return collection, f"SUCCESS: Đã nạp mới {len(ids)} chunks."
        
        else:
            return collection, f"SUCCESS: Dữ liệu đã có sẵn ({collection.count()} chunks)."

    except Exception as e:
        return None, f"ERROR: {str(e)}"

# --- GỌI HÀM NẠP DỮ LIỆU VÀ HIỂN THỊ TRẠNG THÁI ---
with st.spinner("Đang khởi động hệ thống tri thức... (Lần đầu có thể mất 1-2 phút)"):
    collection, status_msg = initialize_database()

if collection is None:
    st.error(f"❌ Lỗi khởi tạo: {status_msg}")
    st.stop()
else:
    # Hiển thị thành công ở sidebar
    st.sidebar.success(f"📦 Dữ liệu: **{collection.count()}** chunks")

# ================= 4. LOGIC TRẢ LỜI CÂU HỎI (RAG) =================
def query_ai(question):
    try:
        # 1. Tìm kiếm dữ liệu liên quan
        results = collection.query(
            query_texts=[question],
            n_results=5, # Lấy 5 đoạn văn bản liên quan nhất
            include=["documents", "metadatas"]
        )

        # 2. Tạo ngữ cảnh (Context)
        context_text = ""
        sources = []
        
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                source_title = meta.get('hierarchy', meta.get('title', 'Thông tin'))
                context_text += f"---\nNguồn: {source_title}\nNội dung: {doc}\n"
                
                # Lưu link nguồn để hiển thị
                url = meta.get('source_url', '')
                if url: sources.append(url)

        if not context_text:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu.", []

        # 3. Gửi cho Gemini
        prompt = f"""
        Bạn là trợ lý ảo hỗ trợ pháp luật Việt Nam.
        Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
        
        YÊU CẦU:
        - Trả lời chính xác, ngắn gọn, dễ hiểu.
        - Nếu là quy trình, hãy liệt kê từng bước.
        - Tuyệt đối không bịa đặt thông tin nếu không có trong ngữ cảnh.
        
        THÔNG TIN THAM KHẢO:
        {context_text}
        
        CÂU HỎI: {question}
        """
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text, list(set(sources))

    except Exception as e:
        return f"Đã xảy ra lỗi khi xử lý: {str(e)}", []

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Tôi có thể giúp gì về thủ tục Thường trú, Tạm trú, Hộ chiếu...?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Xử lý và trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu quy định..."):
            ans, source_links = query_ai(prompt)
            
            # Hiển thị câu trả lời
            st.markdown(ans)
            
            # Hiển thị nguồn tham khảo (nếu có)
            if source_links:
                st.markdown("**🔗 Nguồn tham khảo:**")
                for link in source_links:
                    st.markdown(f"- [{link}]({link})")
            
            # Lưu vào lịch sử chat (chỉ lưu text câu trả lời)
            st.session_state.messages.append({"role": "assistant", "content": ans})
