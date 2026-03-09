import streamlit as st
import requests

# --- 1. UI Configuration ---
st.set_page_config(page_title="CSN RAG Assistant", page_icon="🤖", layout="centered")

st.title("🤖 CSN Technical Test: RAG Bot")
st.info("Ask questions based on the Operational Manual. The bot uses LangGraph Memory for context.")

# --- 2. Initialize Session States ---
# Streamlit reload holeo jate UI er chat history muche na jay
if "messages" not in st.session_state:
    st.session_state.messages = []

# Thread ID for LangGraph MemorySaver
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_001" 

# --- 3. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Chat Input & Processing ---
if prompt := st.chat_input("How can I help you today?"):
    # User message display
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call FastAPI Backend
    with st.spinner("Analyzing document..."):
        try:
            payload = {
                "question": prompt,
                "thread_id": st.session_state.thread_id
            }
            # FastAPI endpoint
            response = requests.post("http://127.0.0.1:8000/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                
                # Assistant message display
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error(f"Backend Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Could not connect to FastAPI: {e}")

# --- 5. Sidebar Options ---
with st.sidebar:
    st.header("Settings")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        # Notun thread_id dile memory fresh hoye jabe
        import uuid
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.write(f"**Session ID:** `{st.session_state.thread_id}`")