# 🏥 Agentic RAG: Biomedical Equipment AI Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.32-red.svg)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-lightgrey.svg)

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed to assist Biomedical Equipment Technicians (BMETs) with complex troubleshooting, preventative maintenance, and safety procedures. 

This application ingests massive, complex OEM service manuals and provides highly accurate, context-aware answers with **exact physical page citations**, strictly refusing to hallucinate when data is missing.

---

## 🚀 The Engineering Challenge

Standard "off-the-shelf" RAG pipelines break down when applied to highly technical, regulated domains like healthcare equipment maintenance. I built this application to solve three specific production walls:

1. **"Page Drift" in Metadata:** Standard PDF loaders use absolute digital indexing (e.g., digital page 120). When manuals have 20-page prefaces, the AI cites "Page 120", but the printed physical book says "Page 100".

2. **Context Bleed:** Querying multiple device manuals simultaneously often results in the AI mixing up maintenance schedules or tool requirements across different machines.

3. **Stateless Troubleshooting:** Real-world hardware repair is a dialogue. If a technician asks "What is Error Code 32?" and then follows up with "How do I fix *it*?", stateless LLMs lose the context.

---

## 🛠️ Architecture & Release History

### v1.1: Custom Ingestion & Zero Context Bleed

Tore out standard LangChain document loaders and engineered a custom data ingestion pipeline.

* **Custom PyMuPDF (`fitz`) Extraction:** Built a parsing loop that actively hunts down the manufacturer's embedded *Logical Page Labels*. The vector database now maps text to the physical printed page.

* **Surgical Metadata Routing:** The FAISS vector store can successfully contrast maintenance schedules across entirely different architectures in a single prompt without cross-contamination.

* **Strict Guardrails:** Prompt-engineered the LCEL pipeline to preserve nested diagnostic lists and completely refuse to guess ("I cannot find this") if a specific part number or procedure isn't explicitly in the manual.

* **Visual Proof: v1.1 Zero Context Bleed & Logical Citations**

![Multi-Device Citations](assets/multi_device_citations_v1.1.png)

### v1.2: Stateful Conversational Memory

Upgraded the AI from a "smart search engine" into a true Agentic Assistant.

* **LangChain Message History:** Implemented `RunnableWithMessageHistory` to wrap the LCEL chain, injecting previous chat logs into the prompt via `MessagesPlaceholder`.

* **Multi-Turn Context Resolution:** The AI successfully resolves pronouns ("it", "that part") across multiple conversational turns, allowing technicians to drill down from high-level errors to deep hardware disassembly steps.

* **Streamlit Session State:** Synchronized the backend LangChain memory buffer with Streamlit's UI rendering loop for a seamless, ChatGPT-style interface.

* **Visual Proof: v1.2 Multi-Turn Memory Resolution**

*The AI successfully remembers the GE Venue Fit R5/R6 context across multiple turns, mapping the pronoun "that procedure" to the correct toolset without hallucinating.*

![GE Venue Fit Context - Turn 1](assets/v1.2_ge_memory_turn1.jpg)

![GE Venue Fit Context - Turn 2](assets/v1.2_ge_memory_turn2.jpg)

---

## 💻 Tech Stack

* **Core Framework:** LangChain (LCEL)

* **LLM:** Google Gemini 2.5 Flash Lite

* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)

* **Vector Database:** FAISS (Facebook AI Similarity Search) - *Configured for local persistence*

* **Document Processing:** PyMuPDF (`fitz`), LangChain `RecursiveCharacterTextSplitter`

* **Frontend:** Streamlit

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**

```bash
git clone [https://github.com/balakrishna-arigala26/ai-engineer-portfolio.git](https://github.com/balakrishna-arigala26/ai-engineer-portfolio.git)
cd ai-engineer-portfolio/biomedical-equipment-ai-assistant
```

---

**2. Create and activate virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

**3. Install Dependencies:**

```bash
pip install -r requirements.txt
```

**4. Set up your API Keys:**

Create a .env file in the project directory and add your Google Gemini API key:

```text
GOOGLE_API_KEY="your_api_key_here"
```

**5. Run the application:**

```bash
streamlit run app.py
```

## 🧪 Testing the Memory Buffer (v1.2)

To verify the multi-turn context resolution without spinning up the UI, a standalone backend test script is included.

```bash
python test_memory.py
```

## 🗺️ Roadmap

v1.3 (Upcoming): Multi-modal RAG. Upgrading the ingestion pipeline to capture embedded schematics and diagrams using vision-capable embedding models, allowing the AI to return visual troubleshooting aids alongside text.
