
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

import llm_configuration
import qna_hr_rag
import data_preparation

llm_configuration = llm_configuration.LLMConfiguration()
lcpp_llm = llm_configuration.prepareLlmInstanceAndGetInstance()

retriever_vectorDB = data_preparation.getVectorDBRetriever()

rag_response_generator = qna_hr_rag.RAGResponseGenerator(retriever_vectorDB, lcpp_llm)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI for Tourism package Customer Acceptance Prediction
st.title("Flykite HR Q&A ChatBot")
st.write("Ask your HR related queries here.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if question := st.chat_input("Ask something"):
    # Display user message in chat message container
    with st.chat_message("User"):
        st.markdown(question)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    response = rag_response_generator.generate_rag_response(question)

    # Display assistant response in chat message container
    with st.chat_message("AI HR"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "AI HR", "content": response})
