import uuid
import streamlit as st
import os
import tempfile
import fitz  # PyMuPDF base library
from dotenv import load_dotenv

# LangChain & Gemini Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Configure the web page
st.set_page_config(page_title="Biomedical AI Assistant", page_icon="🏥", layout="wide")

# --- v1.2 Memory Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "store" not in st.session_state:
    st.session_state.store = {}
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to grab the backend memory
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Configure local directory for persistent vector storage
DB_DIR = "biomedical_faiss_index"

# Cache the embedding model
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Initialize session state and load existing database for Phase 1 Persistence
if "vector_store" not in st.session_state:
    if os.path.exists(DB_DIR):
        st.session_state.vector_store = FAISS.load_local(
            DB_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        st.session_state.vector_store = None


# ==========================================
# --- Sidebar UI (v1.1 Document Ingestion) ---
# ==========================================
st.sidebar.header("Document Ingestion")

# Visual indicator for persistent database load
if st.session_state.vector_store is not None:
    st.sidebar.success("⚡ Persistent Knowledge Base Loaded!")

uploaded_files = st.sidebar.file_uploader(
    "Upload Equipment Manuals (PDF)", 
    type=["pdf"], 
    accept_multiple_files=True
)

# Trigger processing
if uploaded_files and st.sidebar.button("Process Documents"):
    with st.spinner(f"Processing {len(uploaded_files)} manual(s)... Extracting logical page labels."):
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            original_name = uploaded_file.name 
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # --- Automated Logical Page Extraction ---
            documents = []
            pdf_doc = fitz.open(tmp_file_path)
            
            # Attempt to grab the publisher's embedded page labels
            page_labels = pdf_doc.get_page_labels() 

            for i, page in enumerate(pdf_doc):
                text = page.get_text("text")
                
                # Skip entirely blank pages to save vector space
                if not text.strip():
                    continue
                    
                # Automatically assign the true page number
                try:
                    true_page_num = page_labels[i] if page_labels else str(i + 1)
                except IndexError:
                    true_page_num = str(i + 1)

                # Construct the LangChain Document with perfect metadata
                metadata = {
                    "source": original_name,
                    "page": true_page_num 
                }
                documents.append(Document(page_content=text, metadata=metadata))
            
            pdf_doc.close()
            
            # Split the accurately labeled documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
            os.remove(tmp_file_path)

        # Update Vector Store
        if st.session_state.vector_store is None:
            vector_store = FAISS.from_documents(all_chunks, embeddings)
        else:
            new_vector_store = FAISS.from_documents(all_chunks, embeddings)
            st.session_state.vector_store.merge_from(new_vector_store)
            vector_store = st.session_state.vector_store

       # v1.0: Save to disk 
        vector_store.save_local(DB_DIR)
        st.session_state.vector_store = vector_store

    st.sidebar.success(f"✅ Knowledge base updated with exact page mapping!")


# ==========================================
# --- Main UI (v1.2 Conversational Memory) ---
# ==========================================
st.title("🏥 Biomedical Equipment AI Assistant")
st.write("Retrieve troubleshooting steps and safety warnings with precise OEM citations.")

# 1. Draw all previous chat bubbles on the screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Wait for the user to type a new question
if user_input := st.chat_input("Ask a troubleshooting or calibration question..."):
    
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload a manual in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing manuals and chat history..."):
                
                # Initialize LLM and Retriever
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})

                # v1.1: Metadata Extraction formatting
                def format_docs(docs):
                    formatted = []
                    for doc in docs:
                        source = doc.metadata.get("source", "Manual")
                        page = doc.metadata.get("page", "Unknown") 
                        formatted.append(f"--- SOURCE: {source} (Page {page}) ---\n{doc.page_content}")
                    return "\n\n".join(formatted)

                # v1.1 Strict Guardrails
                system_prompt_text = """You are an expert Biomedical Equipment Assistant. Use the retrieved context to answer the technician's question.

                STRICT RULES:
                1. Maintain the original formatting of the manual as closely as possible. 
                2. Use numbered lists for primary steps, and use unnumbered bullet points or indented text for sub-notes. Do not turn sub-notes into new numbered steps.
                3. To keep the UI clean, do NOT put a citation on every single line. Instead, group the citation at the very end of the procedure block if all steps come from the same page (e.g., "Source: [Philips_TC70.pdf, Page 101]").
                4. If the answer is not in the context, say "I cannot find this in the uploaded manuals." Do not guess.

                Context: {context}"""

                # v1.2 Memory Prompt Template
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_text),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}")
                ])

                # Unified LCEL Orchestration: RAG + Memory
                rag_chain = (
                    RunnablePassthrough.assign(
                        context=(lambda x: x["input"]) | retriever | format_docs
                    )
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                memory_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )

                try:
                    # Invoke the chain with the user's input and session ID
                    answer = memory_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        st.error("⚠️ Rate Limit Reached. Please wait 30 seconds.")
                    else:
                        st.error(f"Error: {e}")