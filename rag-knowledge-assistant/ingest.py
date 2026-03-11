from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load the .env file from the parent directory
load_dotenv(dotenv_path="../.env")

print("📄 Loading PDF with PyMuPDF (Layout-Aware)...")
# Using the advanced layout-aware loader
loader = PyMuPDFLoader("docs/git_manual.pdf")
documents = loader.load()

print("✂️  Chunking text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

print("🧠 Generating embeddings and building Vector DB...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("vector_db")

print("✅ Success! PDF processed and stored in FAISS vector database.")