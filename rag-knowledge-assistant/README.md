# RAG Knowledge Assistant


An enterprise-grade Retrieval-Augumented Generation (RAG) pipeline built with Python, LagChain, and the Google Gemini API.


This system ingests local documents, converts them into vector embeddings, stores them in a local FAISS database, and allows users to query the documents using a conversational AI interface. This prevents AI hallucinations by forcing the LLM to ground its answers strictly in the provided context.


## Tech Stack

* **Language:** Python 3
* **Framework:** LangChain
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Embeddings:** Google Generative AI (`text-embedding-004`)
* **LLM:** Google Gemini (`gemini-2.5-flash-lite`)


## Project Architecture

The project split into two distinct processes to mimic real-world enterprise architectures:

1. `ingest.py`: Reads the text files from the `/docs` folder. chunks the text, generates embeddings, and saves the FAISS vector database locally.

2. `main.py`: Loads the vector database, accepts user queries, performs a similarity search, and streams the context to LLM for a grounded response.


## How to Run


**1. Setup Environment**

Ensure you have a `.env` file in the parent directory containing your `GEMINI_API_KEY`.


**2. Install Dependencies**

```bash
pip install -r requirements.txt
```
