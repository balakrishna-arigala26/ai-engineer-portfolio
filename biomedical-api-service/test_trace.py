import os
from dotenv import load_dotenv
load_dotenv() # Load first

from langchain_google_genai import ChatGoogleGenerativeAI

print(f"Tracing enabled: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"Project name: {os.getenv('LANGCHAIN_PROJECT')}")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
print("Sending test message to Gemini...")

# A simple, synchronous call
response = llm.invoke("Hello, this is a test for LangSmith observability. Reply with 'Recieved'.")
print("Response:", response.content)
print("Check your LangSmith dashboard now!")