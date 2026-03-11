# RAG Knowledge Assistant

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline built with Python, LangChain, and the Google Gemini API.

This system ingests complex, multi-column PDF documents, converts them into vector embeddings, stores them in a locally isolated FAISS database, and allows users to query the documents via a CLI interface. This architecture completely prevents AI hallucinations by forcing the LLM to ground its answers strictly in the provided context, refusing to guess if the answer is missing.

## Tech Stack

* **Language:** Python 3
* **Framework:** LangChain
* **Data Ingestion:** PyMuPDF (Layout-Aware Parsing)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Google Generative AI (`gemini-embedding-001`)
* **LLM:** Google Gemini (`gemini-2.5-flash-lite`)

## Project Architecture & Features

The project is split into two distinct processes to mimic real-world enterprise architectures:

1. `ingest.py`: The Data Pipeline. It reads PDF files from the `/docs` directory using PyMuPDF to prevent column scrambling. It then uses a "scalpel" chunking strategy (150 characters) to prevent vector dilution, generates embeddings, and saves the FAISS database locally.
2. `main.py`: The Inference Engine. It loads the vector database, accepts user queries, performs a tuned similarity search (`k=10`), aggregates the context, and sends it to the LLM. 

**X-Ray Debugger:** The `main.py` script includes a custom terminal debugger that prints the exact raw text retrieved by FAISS *before* it hits the LLM, ensuring the data pipeline is functioning perfectly.

## How to Run

**1. Setup Environment**
Ensure you have a `.env` file in the parent directory containing your Google API key:
`GEMINI_API_KEY="your_api_key_here"`

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
