import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_pdf(file_bytes: bytes, filename: str):
    documents = []
    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_labels = pdf_doc.get_page_labels()

    for i, page in enumerate(pdf_doc):
        text = page.get_text("text")
        if not text.strip(): continue
        
        # Extract the logical manufacturer page number
        try:
            true_page_num = page_labels[i] if page_labels else str(i + 1)
        except:
            true_page_num = str(i + 1)

        documents.append(Document(
            page_content=text, 
            metadata={"source": filename, "page": true_page_num}
        ))
    
    pdf_doc.close()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)