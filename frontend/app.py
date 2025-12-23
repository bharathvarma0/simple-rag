import streamlit as st
import requests
import json

# Constants
API_URL = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="F1 2026 Regulations Expert",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessageUser"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="stChatMessageAssistant"] {
        background-color: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🏎️ F1 2026 Regulations Expert")
st.markdown("Ask questions about the new Technical, Sporting, and Financial regulations.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Upload & Ingest"):
            with st.spinner("Uploading and processing..."):
                try:
                    # 1. Upload
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    upload_response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if upload_response.status_code == 200:
                        st.success(f"Uploaded {uploaded_file.name}")
                        
                        # 2. Ingest
                        ingest_response = requests.post(f"{API_URL}/ingest", json={})
                        if ingest_response.status_code == 200:
                            data = ingest_response.json()
                            st.success(f"Ingested {data['num_documents']} documents")
                        else:
                            st.error(f"Ingestion failed: {ingest_response.text}")
                    else:
                        st.error(f"Upload failed: {upload_response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    
    # Ingestion
    st.subheader("Data Ingestion")
    if st.button("Re-ingest All Documents"):
        with st.spinner("Ingesting documents..."):
            try:
                response = requests.post(f"{API_URL}/ingest", json={})
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Ingested {data['num_documents']} documents ({data['num_chunks']} chunks)")
                else:
                    st.error(f"Ingestion failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    
    st.divider()
    
    # Available Documents
    st.subheader("Available Documents")
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [])
            if docs:
                st.write(f"Found {len(docs)} documents:")
                for doc in docs:
                    st.text(f"📄 {doc}")
            else:
                st.info("No documents found.")
        else:
            st.error("Failed to fetch documents.")
    except Exception as e:
        st.error(f"Connection error: {e}")
    
    st.divider()
    
    # Query Settings
    st.subheader("Query Parameters")
    top_k = st.slider("Top-K Documents", min_value=1, max_value=10, value=5)

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**Source:** {source['source']}")
                    st.markdown(f"**Relevance:** {source['similarity_score']:.3f}")
                    st.text(source['preview'])
                    st.divider()

# Chat Input
if prompt := st.chat_input("Ask about the 2026 regulations..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing regulations..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": prompt, "top_k": top_k}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                    
                    # Clean up answer for better math rendering
                    # Replace \[ ... \] with $$ ... $$
                    answer = answer.replace("\\[", "$$").replace("\\]", "$$")
                    # Replace \( ... \) with $ ... $
                    answer = answer.replace("\\(", "$").replace("\\)", "$")
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.markdown(f"**Source:** {source['source']}")
                                st.markdown(f"**Relevance:** {source['similarity_score']:.3f}")
                                st.text(source['preview'])
                                st.divider()
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
