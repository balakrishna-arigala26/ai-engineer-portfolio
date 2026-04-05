import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class BiomedicalAIEngine:
    def __init__(self):
        self.db_dir = "biomedical_faiss_index"
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = {} 
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        # Check if the actual index file exists to prevent empty folder crashes
        index_file = os.path.join(self.db_dir, "index.faiss")
        if os.path.exists(index_file):
            return FAISS.load_local(
                self.db_dir, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        return None

    def update_vector_store(self, chunks):
        """Adds new PDF chunks to the persistent FAISS index."""
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            new_db = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.merge_from(new_db)
        
        self.vector_store.save_local(self.db_dir)

    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    async def ask_question(self, user_input: str, session_id: str):
        if not self.vector_store:
            return "I cannot find any manuals. Please upload a PDF first."

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        def format_docs(docs):
            return "\n\n".join([
                f"--- SOURCE: {d.metadata.get('source')} (Page {d.metadata.get('page')}) ---\n{d.page_content}"
                for d in docs
            ])

        # Strict Guardrails for UI-Perfect Formatting
        template = """You are an expert Biomedical Assistant. Answer based strictly on context.
        
        STRICT RULES FOR FORMATTING:
        1. INTRO: Always start with a brief, natural introductory sentence that directly references the user's question.
        2. STEPS: Immediately follow the intro with a numbered list for the procedures. DO NOT use a heading like "Procedures:".
        3. WARNINGS: If there are safety hazards, leave a blank line, add the exact bold heading '**Safety Warnings:**', and list them using unnumbered bullet points.
        4. NO INLINE CITATIONS: Do not put citations at the end of sentences.
        5. CONSOLIDATED CITATIONS: Place all sources at the VERY END of your response on a single new line. Group pages from the same manual. (Format: Source: [Filename.pdf, Page X, Page Y]).
        6. If unknown, say 'I cannot find this in the manuals.'
        
        Context: {context}"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        rag_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: x["input"]) | retriever | format_docs
            )
            | prompt | llm | StrOutputParser()
        )

        memory_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        return await memory_chain.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )