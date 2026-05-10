from src.research import search_arxiv_papers

from src.chatbot import summarize_research_paper
from src.embeddings import (
    chunk_text,
    create_vector_store,
    save_vector_store,
    load_vector_store
)

from src.chatbot import ask_question

import streamlit as st

from src.pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    save_uploaded_file
)

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="AI Workflow Agent",
    page_icon="🤖",
    layout="wide"
)

# =========================
# SESSION STATE
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.title("🤖 Multi-Agent AI Workflow Platform")

    st.caption(
        "AI-powered document intelligence system using RAG, FAISS, and NVIDIA AI"
    )

    st.markdown("---")

    st.subheader("📌 Project Features")

    st.write("✅ Multi PDF Upload")
    st.write("✅ PDF Processing")
    st.write("✅ Semantic Search")
    st.write("✅ Multi-Agent AI")
    st.write("✅ RAG Pipeline")
    st.write("✅ NVIDIA API Integration")
    st.write("✅ Arxiv Research Search")

    st.markdown("---")

    st.subheader("🧠 Available Agents")

    st.write("📘 Research Agent")
    st.write("📝 Summary Agent")
    st.write("📧 Email Agent")
    st.write("💡 Insight Agent")

    # =========================
    # ARXIV SEARCH
    # =========================
    st.markdown("---")

    st.subheader("🔬 AI Research Search")

    research_query = st.text_input(
        "Search Research Papers",
        placeholder="Example: Large Language Models"
    )

# =========================
# MAIN TITLE
# =========================
st.title("📄 AI Workflow Agent")

st.write("Upload PDF files and interact with AI agents.")

# =========================
# FILE UPLOADER
# =========================
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================
# MAIN WORKFLOW
# =========================
if uploaded_files:

    st.success("✅ PDFs uploaded successfully!")

    all_text = ""
    pdf_names = []

    # Process all uploaded PDFs
    for uploaded_file in uploaded_files:

        pdf_names.append(uploaded_file.name)

        # Save PDF
        saved_path = save_uploaded_file(uploaded_file)

        st.info(f"📁 File saved at: {saved_path}")

        # Extract text
        raw_text = extract_text_from_pdf(uploaded_file)

        # Clean text
        cleaned_text = clean_text(raw_text)

        # Combine all PDF text
        all_text += f"\n\nDOCUMENT: {uploaded_file.name}\n\n"
        all_text += cleaned_text

    # =========================
    # CREATE CHUNKS
    # =========================
    chunks = chunk_text(all_text)

    # =========================
    # METRICS DASHBOARD
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Chunks", len(chunks))

    with col2:
        st.metric("📚 PDFs", len(uploaded_files))

    with col3:
        st.metric("🧠 AI Agents", 4)

    st.markdown("---")

    # =========================
    # UPLOADED DOCUMENTS
    # =========================
    st.subheader("📂 Uploaded Documents")

    for pdf in pdf_names:
        st.write(f"✅ {pdf}")

    st.markdown("---")

    # =========================
    # CREATE VECTOR STORE
    # =========================
    vector_store = create_vector_store(chunks)

    save_vector_store(vector_store)

    st.success("✅ Vector database created successfully!")

    # =========================
    # EXPANDABLE CHUNK VIEWER
    # =========================
    with st.expander("📦 View Text Chunks"):

        for i, chunk in enumerate(chunks[:5]):

            st.write(f"### Chunk {i+1}")
            st.write(chunk)
            st.markdown("---")

    # =========================
    # EXTRACTED TEXT
    # =========================
    st.subheader("📚 Extracted & Cleaned Text")

    st.text_area(
        "PDF Content",
        all_text,
        height=300
    )

    st.markdown("---")

    # =========================
    # AI AGENT SELECTOR
    # =========================
    st.subheader("🧠 Select AI Agent")

    agent_type = st.selectbox(
        "Choose an Agent",
        [
            "Research Agent",
            "Summary Agent",
            "Email Agent",
            "Insight Agent"
        ]
    )

    # Agent Descriptions
    if agent_type == "Research Agent":
        st.info(
            "📘 Best for technical explanations and detailed analysis."
        )

    elif agent_type == "Summary Agent":
        st.info(
            "📝 Best for concise summaries and simplified explanations."
        )

    elif agent_type == "Email Agent":
        st.info(
            "📧 Generates professional email drafts from document context."
        )

    elif agent_type == "Insight Agent":
        st.info(
            "💡 Extracts trends, insights, and observations."
        )

    st.markdown("---")

    # =========================
    # CHATBOT SECTION
    # =========================
    st.subheader("💬 Chat with Your PDFs")

    user_question = st.text_area(
        "💬 Ask Questions About Your Documents",
        placeholder=(
            "Example: Summarize the uploaded PDFs or explain key findings..."
        ),
        height=120
    )

    # =========================
    # AI RESPONSE BUTTON
    # =========================
    if st.button("🚀 Generate AI Response"):

        if user_question:

            with st.spinner(
                "AI is analyzing your documents..."
            ):

                # Load vector DB
                vector_store = load_vector_store()

                # Generate answer
                answer = ask_question(
                    vector_store,
                    user_question,
                    agent_type
                )

            # Save chat history
            st.session_state.chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                    "agent": agent_type
                }
            )

    # =========================
    # DISPLAY CHAT HISTORY
    # =========================
    if st.session_state.chat_history:

        st.markdown("---")

        st.subheader("📜 Conversation History")

        for chat in reversed(st.session_state.chat_history):

            with st.container():

                st.markdown(f"""
### 🧑 User Question
{chat['question']}

### 🤖 {chat['agent']}
{chat['answer']}
""")

                st.markdown("---")

else:

    st.info("📄 Upload PDFs to start the AI workflow.")

    st.markdown("""
## 🔄 Workflow Pipeline

PDF Upload → Text Extraction → Cleaning → Embeddings → FAISS Vector DB → Multi-Agent AI Chatbot
""")

# =========================
# ARXIV RESEARCH SECTION
# =========================
if research_query:

    st.markdown("---")

    st.subheader("📚 Research Papers")

    papers = search_arxiv_papers(research_query)

    for paper in papers:

        with st.container():

            st.markdown(f"## {paper['title']}")

            st.write(
                f"👨‍🔬 Authors: {', '.join(paper['authors'])}"
            )

            st.write("### Abstract")
            st.write(paper["summary"])

            with st.spinner("Generating AI Summary..."):

                ai_summary = summarize_research_paper(
                    paper["summary"]
                )

            st.write("### 🤖 AI Summary")
            st.write(ai_summary)

            st.markdown(
                f"[📄 Read Paper]({paper['pdf_url']})"
            )

            st.markdown("---")
            
            st.metric(
                "📄 Papers Found",
                len(papers)
            )

# =========================
# FOOTER
# =========================
st.markdown("---")

st.caption(
    "Built using Streamlit, LangChain, FAISS, NVIDIA AI, and Arxiv API"
)