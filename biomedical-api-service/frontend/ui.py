import streamlit as st
import requests
import uuid
import time  # <-- 1. NEW IMPORT ADDED HERE

# Setup the page to look professional
st.set_page_config(page_title="Enterprise Biomedical AI", page_icon="🛠️")
st.title("🛠️ Enterprise Biomedical AI Assistant")
st.caption("Agentic RAG for Clinical Hardware Troubleshooting")

# Create a unique session ID for the backend memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Store the chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat bubbles
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# The Chat Input Box
if prompt := st.chat_input("Ex: How do I replace the battery on Venue Fit?"):

    # 1. Show the user's question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get the AI's answer
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                "http://localhost:8000/ask-stream",
                json={"question": prompt, "session_id": st.session_state.session_id},
                stream=True
            )
            
            # --- 2. THE NEW SMOOTHER FUNCTION ---
            def get_stream():
                # Read larger chunks from the network so it doesn't bottleneck
                for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                    if chunk:
                        # Break the chunky response into characters to force a smooth animation
                        for char in chunk:
                            yield char
                            time.sleep(0.005) # 5-millisecond delay per character

            # Streamlit animates the text beautifully
            full_answer = st.write_stream(get_stream())
            st.session_state.messages.append({"role": "assistant", "content": full_answer})
            
        except Exception as e:
            st.error(f"Connection failed: {e}")