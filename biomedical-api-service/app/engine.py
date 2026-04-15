import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class BiomedicalAIEngine:
    def __init__(self):
        self.db_dir = "chroma_db" 
        
        # Keeping your secure, local embeddings for data privacy
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.store = {} 
        
        # Chroma automatically loads from disk if it exists, or creates it if it doesn't
        self.vector_store = Chroma(
            persist_directory=self.db_dir,       
            embedding_function=self.embeddings   
        ) 
        
        # Define the LLM here so it's ready for the whole class
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1, 
            streaming=True 
        )

    
    def update_vector_store(self, chunks):
        """Adds new PDF chunks to the persistent ChromaDB index."""
        self.vector_store.add_documents(chunks)

    
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    
    def _get_chain(self):
        """Builds the conversational RAG chain with your strict formatting"""

        # Check if the database is empty
        if self.vector_store._collection.count() == 0:
            raise ValueError("No database found. Please upload manuals first.")
        
        
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        template = """You are an expert Biomedical Assistant. You are provided with Context below. The Context contains snippets of medical manuals.

        The Context will include source tags that look like . DO NOT output these tags in the middle of your sentences.

        STRICT RULES FOR FORMATTING:
        1. INTRO: Always start with a brief, natural introductory sentence that directly references the user's question.
        2. STEPS: Immediately follow the intro with a numbered list for the procedures. DO NOT use a heading like "Procedures:".
        3. WARNINGS: If there are safety hazards, leave a blank line, add the exact bold heading '**Safety Warnings:**', and list them using unnumbered bullet points.
        4. NO INLINE CITATIONS: NEVER put tags inside your sentences.
        5. CONSOLIDATED CITATIONS: Place all sources at the VERY END of your response on a single new line formatted exactly like this: "Source: [Manual Name, Page X]". (You will have to infer the Manual Name and Page number from the context).
        6. If unknown, say 'I cannot find this in the manuals.'
        
        Context: {context}"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Safely construct the LangChain pipeline
        rag_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: x["input"]) | retriever | format_docs
            )
            | prompt 
            | self.llm 
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    # --- THE TWO WAYS TO ASK QUESTIONS ---

    async def ask_question(self, question: str, session_id: str):
        """Standard method: Waits for the whole answer before returning."""
        chain = self._get_chain()
        return await chain.ainvoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        )

    async def stream_question(self, question: str, session_id: str):
        """Streaming method: Yields text chunk-by-chunk for the UI."""
        chain = self._get_chain()
        
        async for chunk in chain.astream(
            {"input": question},
            config={"configurable": {"session_id": session_id}}
        ):
            # Send chunks to the frontend instantly
            if isinstance(chunk, str):
                yield chunk
            elif hasattr(chunk, 'content'):
                yield chunk.content