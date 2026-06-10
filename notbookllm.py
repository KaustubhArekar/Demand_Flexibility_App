# notebooklm_style.py - RAG-based Document Analysis
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pathlib import Path
import pickle

class DocumentQASystem:
    """Local RAG system for document analysis (NotebookLM-style) [citation:10]"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.chunks = []
        self.source_docs = {}
    
    def load_documents(self, files):
        """Load and chunk uploaded documents"""
        self.chunks = []
        for file in files:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
                # Convert DataFrame to text chunks
                for col in df.columns:
                    summary = f"Column {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
                    self.chunks.append({
                        "text": summary,
                        "source": file.name,
                        "type": "summary"
                    })
                    
                    # Add sample rows as chunks
                    sample_text = df.head(10).to_string()
                    self.chunks.append({
                        "text": sample_text,
                        "source": file.name,
                        "type": "data_sample"
                    })
        
        # Build FAISS index
        embeddings = self.embedding_model.encode([c["text"] for c in self.chunks])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))
        
        return len(self.chunks)
    
    def query(self, question, top_k=3):
        """Retrieve relevant chunks and generate answer"""
        if self.index is None:
            return "No documents loaded. Please upload files first."
        
        # Search for relevant chunks
        question_embedding = self.embedding_model.encode([question])
        distances, indices = self.index.search(np.array(question_embedding).astype('float32'), top_k)
        
        # Get relevant chunks
        relevant_chunks = [self.chunks[i] for i in indices[0]]
        
        # Format context
        context = "\n\n".join([f"[{c['source']}]: {c['text']}" for c in relevant_chunks])
        
        return context, relevant_chunks

# Streamlit UI for NotebookLM-style analysis
def notebooklm_tab():
    st.subheader("📚 Document Analysis (NotebookLM Style)")
    
    # Initialize RAG system
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = DocumentQASystem()
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload optimization documents (CSV, Excel, TXT)",
        accept_multiple_files=True,
        type=["csv", "xlsx", "txt"]
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner(f"Processing {len(uploaded_files)} files..."):
            num_chunks = st.session_state.rag_system.load_documents(uploaded_files)
            st.success(f"✅ Processed {num_chunks} document chunks")
    
    # Display source documents
    if st.session_state.rag_system.chunks:
        with st.expander("📄 Document Sources"):
            sources = set(c["source"] for c in st.session_state.rag_system.chunks)
            for src in sources:
                chunk_count = sum(1 for c in st.session_state.rag_system.chunks if c["source"] == src)
                st.write(f"- {src}: {chunk_count} chunks")
    
    # Chat interface
    st.markdown("### 💬 Ask about your documents")
    
    for msg in st.session_state.rag_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📎 Sources"):
                    for src in msg["sources"]:
                        st.caption(f"📄 {src}")
    
    query = st.chat_input("Ask a question about your optimization data...")
    
    if query:
        st.session_state.rag_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                context, sources = st.session_state.rag_system.query(query)
                
                # Generate response using DeepSeek
                response_prompt = f"""Based on the following document context, answer the user's question:
                
                Context:
                {context}
                
                Question: {query}
                
                Answer concisely with specific insights from the data."""
                
                messages = [{"role": "user", "content": response_prompt}]
                response = get_deepseek_response(messages, "You are a data analysis expert.")
                
                st.markdown(response)
                with st.expander("📎 Sources used"):
                    for src in sources:
                        st.caption(f"📄 {src['source']}: {src['text'][:200]}...")
        
        st.session_state.rag_messages.append({
            "role": "assistant", 
            "content": response,
            "sources": [s["source"] for s in sources]
        })