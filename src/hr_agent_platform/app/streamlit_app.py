import streamlit as st
import uuid
import sys
import os

# Add src to sys.path to import our platform
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from hr_agent_platform.scripts.graphs import graph
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="HR Agent Platform", page_icon="🤖", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 HR Intelligent Assistant")
st.markdown("---")

# Sidebar for session settings
with st.sidebar:
    st.header("Session Settings")
    uid = st.text_input("Employee ID", value="EMP001", help="Enter your unique Employee ID")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.info("This agent helps with HR Policies, Leave Applications, and Travel Booking.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.spinner("Thinking..."):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "uid": uid
        }
        
        try:
            result = graph.invoke(initial_state, config=config)
            response = result.get("response", "I encountered an error processing your request.")
            
            # Show response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"Error: {e}")

# Footer
st.markdown("---")
