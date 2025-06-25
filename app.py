import streamlit as st
from indexmanager import IndexManager
from agent import Agent
from constants import embed_model, llm_model

@st.cache_resource
def initialize_agent():
    """
    Initialize the agent with the index and LLM model.
    
    Returns:
        Agent: An instance of the Agent class.
    """
    index_manager = IndexManager(embed_model)
    index = index_manager.retrieve_index()
    return Agent(index, llm_model)

if "agent" not in st.session_state:
    st.session_state.agent = initialize_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Research Paper Chatbot")

#display chat messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about recent research papers"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt)
            st.markdown(response.response)
            st.session_state.messages.append({"role": "assistant", "content": response.response})