from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import sys

# Load the .env file from the parent directory
load_dotenv(dotenv_path="../.env")

try:
    # 1. Load the Embedding model and the local FAISS database
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

    # 2. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

    print("RAG Knowledge Assistant (Powered by Gemini & FAISS)")
    print("Type 'exit' or 'quit' to quit\n")
    print("-" * 60)

    while True:
        query = input("\n🧑 Ask: ")

        if query.lower() in ["exit", "quit"]:
            print("\n🤖 AI: Goodbye!")
            break
            
        if not query.strip():
            continue

        # 3. Perform a Similarity Search (Grabbing the top 4 chunks instead of 1)
        docs = db.similarity_search(query, k=10)
        
        # Extract and combine the text from ALL relevant chunks
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant information found."

        # --- THE X-RAY DEBUGGER ---
        print("\n[DEBUG] Here is the exact text FAISS found in the PDF:")
        print("*" * 40)
        print(context[:600] + "...\n") # Printing the first 600 characters
        print("*" * 40)
        # --------------------------

        # 4. Construct the prompt with the retrieved context
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context. 
        If the answer is not contained in the context, say "I don't know based on the provided document."
        
        Context:
        {context}

        Question:
        {query}
        """

        # 5. Send the prompt to Gemini and print the response
        response = llm.invoke(prompt)
        print(f"\n🤖 AI:\n{response.content}")
        print("-" * 60)

except Exception as e:
    print(f"\n❌ A critical error occurred: {e}")