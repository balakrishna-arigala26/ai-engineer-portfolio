import streamlit as st
import os
import tempfile 
from dotenv import load_dotenv 

# LangChain & Gemini Imports 
from langchain_community.document_loaders import PyMuPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_community.vectorstores import FAISS 
from langchain_core.prompts import PromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configure the web page 
st.set_page_config(page_title="Biomedical AI Assistant", page_icon="🏥", layout="centered") 

# Configure local directory for persistent vector storage
DB_DIR = "biomedical_faiss_index"

# Cache the embedding model to prevent reloading on UI intersections
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Initialize session state and load existing database if present for zero-latency boot
if "vector_store" not in st.session_state:
    if os.path.exists(DB_DIR):
        st.session_state.vector_store =FAISS.load_local(
            DB_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("🏥 Biomedical Equipment AI Assistant")
st.write("Upload OEM service manuals to instantly retrieve troubleshooting steps, safety warnings, and calibration procedures.")

# Sidebar UI (Multi-Document Ingestion)
st.sidebar.header("Document Ingestion")

# Visual indicator for successful database load
if st.session_state.vector_store is not None:
    st.sidebar.success("⚡ Persistent Knowledge Base Loaded instantly!")
    st.sidebar.info("Uploaded more PDFs to add them to the existing database.")

uploaded_files = st.sidebar.file_uploader(
    "Upload Equipment Manuals (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

# Trigger processing
if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner(f"Processing {len(uploaded_files)} manual(s)... This may take a minute."):
        all_chunks = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyMuPDFLoader(tmp_file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            all_chunks.extend(chunks)
            os.remove(tmp_file_path)

        # Initialize or update the vector database
        if st.session_state.vector_store is None:
            vector_store = FAISS.from_documents(all_chunks, embeddings)
        else:
            new_vector_store = FAISS.from_documents(all_chunks, embeddings)
            st.session_state.vector_store.merge_from(new_vector_store)
            vector_store = st.session_state.vector_store

        # Save vector database locally to hard drive
        vector_store.save_local(DB_DIR)
        st.session_state.vector_store = vector_store

    st.sidebar.success(f"✅ Knowledge base built and saved to disk!")
    
# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt :=st.chat_input("Ask a troubleshooting or calibration question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Validate database state before querying
    if st.session_state.vector_store is None:
        warning = "⚠️ Please upload and process at least one equipment manual in the sidebar first."

        with st.chat_message("assistant"):
            st.markdown(warning)
        st.session_state.messages.append({"role": "assistant", "content": warning})
    else:
        # Generate AI Response
        with st.chat_message("assistant"):
            with st.spinner("Searching technical manuals..."):
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})

                # Strct Biomedical System Prompt with hallucination guardrails
                template = """You are an expert Biomedical Equipment Assistant. Use th following pieces of retrieved service manual context to answer the technician's question.
                If the answer is not ins not in the manual, say "I canot find this in the uploaded manuals." Do not guess.
                
                Context: {context}
                
                Question: {question}
                
                Answer:"""
                custom_prompt = PromptTemplate.from_template(template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content  for doc in docs)
                
                # LangChain Expression Language(LCEL) orchestration
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | custom_prompt
                    | llm
                    | StrOutputParser()
                )
                
                try:
                    answer = rag_chain.invoke(prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        error_msg = "⚠️ **API Rate Limit Reached:** We are using the free tier of the AI model. Please wait 30 seconds and try your question again."
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    else:
                        st.error(f"An unexpected error occurred: {e}")