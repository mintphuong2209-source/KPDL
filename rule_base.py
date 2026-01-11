# ================= 1. FIX LỖI SQLITE (BẮT BUỘC ĐỂ TRÊN CÙNG) =================
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# =============================================================================

import streamlit as st
import json
import os
import glob
import uuid
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Chatbot Hỗ Trợ Cư Trú", layout="wide")
st.title("🤖 Chatbot Tư Vấn Thủ Tục Cư Trú")
COLLECTION_NAME = "dichvucong_multi_files"

# ================= 2. CẤU HÌNH API & MODEL =================
st.sidebar.header("⚙️ Cấu hình")

api_key = st.secrets.get("GEMINI_API_KEY") 
if not api_key:
    api_key = st.sidebar.text_input("Nhập Google AI Studio API Key:", type="password")
    if not api_key:
        st.warning("👉 Vui lòng nhập API Key để bắt đầu.")
        st.stop()

genai.configure(api_key=api_key)

# Tự động tìm model Gemini tốt nhất
try:
    available_models = [m.name for m in genai.list_models() if 'gemini' in m.name and 'generateContent' in m.supported_generation_methods]
    if available_models:
        # Ưu tiên Flash hoặc Pro
        default_idx = 0
        for i, m in enumerate(available_models):
            if "flash" in m: default_idx = i; break
        SELECTED_MODEL = st.sidebar.selectbox("Chọn Model AI:", available_models, index=default_idx)
    else:
        st.sidebar.error("❌ Không tìm thấy model Gemini nào.")
        st.stop()
except Exception as e:
    st.sidebar.error(f"Lỗi API: {e}")
    st.stop()

# ================= 3. HÀM LOAD TẤT CẢ FILE JSON =================
@st.cache_resource(ttl="2h") 
def initialize_vector_db():
    EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
    
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
        
        # Chỉ nạp nếu DB rỗng
        if collection.count() == 0:
            
            # 1. Tìm tất cả file .json
            list_json_files = glob.glob("*.json")
            if not list_json_files:
                st.error("⚠️ Không tìm thấy file .json nào trên GitHub!")
                return None
            
            st.toast(f"Đang xử lý {len(list_json_files)} file dữ liệu...", icon="📂")
            
            all_data = []
            seen_content = set() # Dùng để lọc trùng
            
            # 2. Đọc và gộp file
            file_report = []
            for file_name in list_json_files:
                try:
                    with open(file_name, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # Lọc trùng lặp ngay khi đọc
                            count_added = 0
                            for item in data:
                                content = item.get("content_text", "").strip()
                                # Chỉ thêm nếu nội dung chưa từng xuất hiện và đủ dài
                                if content and content not in seen_content and len(content) > 5:
                                    seen_content.add(content)
                                    all_data.append(item)
                                    count_added += 1
                            file_report.append(f"✅ {file_name}: lấy {count_added} chunks")
                        else:
                            file_report.append(f"⚠️ {file_name}: Sai định dạng")
                except:
                    file_report.append(f"❌ {file_name}: Lỗi đọc")

            # Hiển thị báo cáo file
            with st.sidebar.expander("📄 Chi tiết nạp dữ liệu"):
                for line in file_report:
                    if "✅" in line: st.success(line)
                    else: st.warning(line)

            if not all_data:
                st.error("Không có dữ liệu hợp lệ để nạp!")
                return None

            st.toast(f"Tổng hợp được {len(all_data)} chunks sạch (đã lọc trùng). Đang nạp...", icon="🚀")

            # 3. Chuẩn bị dữ liệu nạp vào Chroma
            # Tạo ID mới hoàn toàn để tránh lỗi trùng ID cũ
            ids = [str(uuid.uuid4()) for _ in all_data]
            documents = [item["content_text"] for item in all_data]
            metadatas = []
            
            for item in all_data:
                meta = item.get("metadata", {}).copy()
                meta.update({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "hierarchy": item.get("hierarchy", ""),
                })
                clean_meta = {k: (str(v) if v is not None else "") for k, v in meta.items()}
                metadatas.append(clean_meta)
            
            # 4. Nạp Batch (40 items/lần)
            batch_size = 40
            progress_bar = st.sidebar.progress(0)
            
            for i in range(0, len(ids), batch_size):
                collection.add(
                    ids=ids[i:i+batch_size],
                    documents=documents[i:i+batch_size],
                    metadatas=metadatas[i:i+batch_size]
                )
                progress_bar.progress(min((i + batch_size) / len(ids), 1.0))
            
            progress_bar.empty()
            st.toast(f"Hoàn tất! Đã nạp {len(ids)} chunks.", icon="🎉")
                
        return collection
        
    except Exception as e:
        st.error(f"Lỗi khởi tạo DB: {str(e)}")
        return None

# Gọi hàm khởi tạo
collection = initialize_vector_db()

if not collection:
    st.stop()
    
# HIỆN TRẠNG THÁI
st.sidebar.success(f"📦 Dữ liệu: **{collection.count()}** chunks (Đã lọc trùng)")

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
                context_parts.append(f"--- {h} ---\n{doc}")
                sources.append(f"- [{h}]({u})")
                
        context = "\n\n".join(context_parts)
        
        prompt = f"""
        Bạn là trợ lý ảo hành chính công. 
        Hãy trả lời câu hỏi dựa trên thông tin được cung cấp dưới đây.
        
        YÊU CẦU:
        - Trả lời chi tiết, rõ ràng, đúng trọng tâm.
        - Nếu là quy trình, hãy liệt kê từng bước.
        - Không tự bịa đặt thông tin.
        
        NGỮ CẢNH:
        {context}
        
        CÂU HỎI: {query_text}
        """
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text, list(set(sources)), context
        
    except Exception as e:
        return f"Lỗi xử lý: {str(e)}", [], ""

# ================= 5. GIAO DIỆN CHAT =================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Xin chào! Bạn cần tìm hiểu thủ tục gì?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu..."):
            answer, sources, debug_ctx = query_rag(prompt, SELECTED_MODEL)
            
            if sources:
                full_resp = f"{answer}\n\n**📚 Nguồn tham khảo:**\n" + "\n".join(sources)
            else:
                full_resp = answer
            
            st.markdown(full_resp)
            
            with st.expander("🕵️ Xem dữ liệu tìm được"):
                st.text(debug_ctx)
            
            st.session_state.messages.append({"role": "assistant", "content": full_resp})
