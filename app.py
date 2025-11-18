import streamlit as st
from main_agent import AgentInterface
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title='Specialized LLM Agent', layout='wide')

st.title("LAU Chatbot")

st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload documents for RAG (PDF/TXT)", accept_multiple_files=True)
if uploaded:
    st.sidebar.write("Uploaded files will be used for RAG ingestion on the server side (not automatic in this template).")

st.sidebar.markdown("**API usage**: Remember to set your API keys in the `.env` file.")

agent = AgentInterface()

st.header("Chat")
with st.form("chat_form"):
    user_input = st.text_area("Enter your question", height=120)
    submitted = st.form_submit_button("Send")
if submitted and user_input.strip():
    with st.spinner("Thinking..."):
        response, metadata = agent.run_query(user_input)
    st.subheader("Response")
    st.write(response)
    if metadata:
        st.markdown("**Retrieval context / tools**")
        st.json(metadata)

st.header("Tools")
if st.button("Run example tool: keyword extractor on sample text"):
    sample = "This project implements 7oto hone shu bedkon."
    keywords = agent.keyword_extractor_tool(sample)
    st.write("Keywords:", keywords)

st.header("Mental Health Support (Non-clinical)")

with st.form("mood_support_form"):
    user_feelings = st.text_area(
        "Describe how you're feeling (entirely optional — this tool offers gentle, non-clinical emotional support):",
        height=120
    )
    mood_submitted = st.form_submit_button("Get supportive message")

if mood_submitted and user_feelings.strip():
    support = agent.mood_support_tool(user_feelings)
    st.subheader("Supportive Message")
    st.write(support)