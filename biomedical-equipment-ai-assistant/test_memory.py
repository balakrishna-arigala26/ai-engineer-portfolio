import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory 

# Load your existing GOOGLE_API_KEY from .env
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Set up the Prompt with the new Memory Placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert Biomedical Equipment Assistant. Answer concisely."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the base LCEL chain
chain = prompt | llm

# Set up the session hsitory dictionary (The "Brain")
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap the chain with the Memory Logic
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    inpu_messages_key="input",
    history_messages_key="chat_history"
)

# =========================================================
# TERMINAL TESTING
# =========================================================
print("\n--- Initiating v1.2 Memory Test --- \n")

# Turn 1: Providing initial context
print("User: I am working on Siemens Mobilett Plus E. It is throwing Error Code 32.")
response1 = memory_chain.invoke(
    {"input": "I am working on a Siemens Mobilett Plus E. It is throwing Error Code 32."},
    config={"configurable": {"session_id": "session_1"}}
)
print(f"AI: {response1.content}\n")

# Turn 2: Testing the memory (No mention of Siemens or Error 32)
print("User: What are the safety precautions before I open it up to fix that error?")
response2 = memory_chain.invoke(
    {"input": "What are the safety precaustions before I open it up to fix that error?"},
    config={"configurable": {"session_id": "session_1"}}
)
print(f"AI: {response2.content}\n")
