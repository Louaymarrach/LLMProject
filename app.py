"""
app.py - Streamlit UI for the LAU Chatbot and tools.

Place your CSS in `style.css` next to this file (already provided the styling).
This app expects:
 - tools.py in the same folder (Toolset)
 - main_agent.py (AgentInterface) in the same folder
 - .env configured for any API keys (Google generative API, etc.)
"""

import os
import html  # For escaping user input
import streamlit as st
from dotenv import load_dotenv
from main_agent import AgentInterface

load_dotenv()

st.set_page_config(page_title="LAU Chatbot", layout="wide", initial_sidebar_state="auto")

# Load local CSS (style.css must exist in same directory)
def local_css(file_name: str):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Small fallback style if file not found
        st.markdown(
            "<style>"
            "body{font-family: 'Segoe UI', sans-serif; background-color:#f2f4f7}"
            ".section-card{background:white;border-radius:6px;padding:16px;margin-bottom:12px;box-shadow:0 1px 3px rgba(0,0,0,0.08)}"
            "</style>",
            unsafe_allow_html=True,
        )


local_css("style.css")

# Title
st.title("LAU Chatbot")

# Sidebar
st.sidebar.header("Settings & Tools")
uploaded = st.sidebar.file_uploader("Upload documents for RAG (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"])
st.sidebar.markdown("**Note:** Uploaded files will be saved to the `uploads/` folder and ingested when you click the Ingest button below.")
st.sidebar.markdown("---")
st.sidebar.markdown("**API usage**: Ensure your API keys (e.g. Google) are set in `.env`.")

# Ensure uploads dir exists
os.makedirs("uploads", exist_ok=True)

# Instantiate agent (this may take a moment depending on your LLM initialization)
with st.spinner("Initializing agent..."):
    agent = AgentInterface()

# ----------------- Layout -----------------
col1, col2 = st.columns([2, 1])

# ----------------- Left: Chat -----------------
with col1:
    st.markdown('<div class="section-card blue">', unsafe_allow_html=True)
    st.header("Chat")

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Function to send a message
    def send_message():
        user_input = st.session_state.input_text
        if not user_input.strip():
            return
        response, _ = agent.run_query(user_input)  # metadata ignored here
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "agent", "message": response})
        st.session_state.input_text = ""  # clear input after sending

    # Display chat history
    for chat in st.session_state.chat_history:
        msg_safe = html.escape(chat["message"])
        if chat["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align:right; margin-bottom:8px;'>
                    <div style='display:inline-block; background:#DCF8C6; padding:10px 14px; border-radius:12px; max-width:80%; word-wrap: break-word;'>
                        {msg_safe}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='text-align:left; margin-bottom:8px;'>
                    <div style='display:inline-block; background:#F1F0F0; padding:10px 14px; border-radius:12px; max-width:80%; word-wrap: break-word;'>
                        {msg_safe}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Chat input area below chat history
    st.text_area("Enter your question", key="input_text", height=140)
    st.button("Send", on_click=send_message)

    st.markdown("</div>", unsafe_allow_html=True)

    # RAG QA specific interface
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("Ask documents (RAG) — uses ingested docs")
    rag_q = st.text_input("Question for documents (RAG):")
    rag_k = st.slider("Number of context chunks to retrieve (k)", 1, 6, 3)
    if st.button("Ask documents"):
        if not os.path.exists("vectorstore") and agent.retriever is None:
            st.warning("No vectorstore loaded — ingest documents first in the sidebar.")
        else:
            with st.spinner("Retrieving and answering..."):
                out = agent.pdf_qa_tool(rag_q, k=rag_k)
            if out.get("status") == "ok":
                st.subheader("Answer")
                st.write(out.get("answer"))
                st.markdown("**Top sources**")
                st.json(out.get("sources", []))
            else:
                st.error(out.get("error", "Unknown error from pdf_qa"))
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Right: Tools & Uploads -----------------
with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("Tools")

    # Keyword example
    if st.button("Run example: extract keywords from sample"):
        sample = "This project implements a search-and-summarize pipeline for university documents and tools."
        kws = agent.keyword_extractor_tool(sample)
        st.write("Keywords:", kws)

    st.markdown("---")

    # Mood support
    st.header("Mental Health Support (Non-clinical)")
    with st.form("mood_support_form"):
        mood_text = st.text_area("Describe how you're feeling:", height=120)
        mood_btn = st.form_submit_button("Get supportive message")
    if mood_btn and mood_text.strip():
        support_msg, mmeta = agent.mood_support_tool(mood_text)
        st.subheader("Supportive Message")
        st.write(support_msg)
        if mmeta:
            st.json(mmeta)

    st.markdown("</div>", unsafe_allow_html=True)

    # File upload & ingestion UI
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.header("Document ingestion & summarizer")
    st.write("Upload PDFs/TXT and click *Save uploads* then *Ingest into vectorstore* to make them available for RAG.")
    files = st.file_uploader("Upload files (PDF/TXT)", accept_multiple_files=True, type=["pdf", "txt"], key="uploader2")
    if files:
        st.write("Files ready to save.")
        if st.button("Save uploaded files"):
            saved_paths = []
            for f in files:
                save_path = os.path.join("uploads", f.name)
                with open(save_path, "wb") as fh:
                    fh.write(f.getbuffer())
                saved_paths.append(save_path)
            st.success(f"Saved {len(saved_paths)} files to uploads/")
            st.session_state.setdefault("last_saved_uploads", [])
            st.session_state["last_saved_uploads"] = saved_paths

        if st.button("Ingest into vectorstore (RAG)"):
            to_ingest = st.session_state.get("last_saved_uploads", [])
            if not to_ingest:
                st.warning("No saved uploads found. Click 'Save uploaded files' first.")
            else:
                with st.spinner("Ingesting..."):
                    result = agent.ingest_tool(to_ingest)
                if result.get("status") == "ok":
                    st.success("Ingestion finished.")
                    st.write(result.get("report", {}))
                else:
                    st.error(result.get("error", "Ingestion failed"))
                    st.json(result)

    st.markdown("---")
    # Summarize single file
    st.write("Summarize a single saved file (quick TL;DR). Choose a file from uploads.")
    uploads = sorted(os.listdir("uploads")) if os.path.exists("uploads") else []
    sel = st.selectbox("Choose a saved file", options=["(none)"] + uploads)
    max_chars = st.slider("Max characters to include in prompt excerpt", 200, 5000, 1200)
    if st.button("Summarize selected file"):
        if sel == "(none)":
            st.warning("No file selected.")
        else:
            path = os.path.join("uploads", sel)
            with st.spinner("Summarizing..."):
                out = agent.summarize_tool(path, max_chars=max_chars)
            if out.get("status") == "ok":
                st.subheader("Summary")
                st.write(out.get("summary"))
            else:
                st.error(out.get("error", "Summarization failed"))
                st.json(out)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer / small tips
st.markdown(
    "<div style='margin-top:18px;color:#6b7280'>Tip: after ingesting, you can ask document-specific questions with the 'Ask documents (RAG)' box.</div>",
    unsafe_allow_html=True,
)
