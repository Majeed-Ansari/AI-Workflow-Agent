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

# Page Configuration
st.set_page_config(
    page_title="AI Workflow Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# SIDEBAR
with st.sidebar:

    st.title("🤖 Multi-Agent AI Workflow Platform")

    st.caption(
        "AI-powered document intelligence system using RAG, FAISS, and NVIDIA AI"
    )

    st.markdown("---")

    st.subheader("📌 Project Features")

    st.write("✅ PDF Processing")
    st.write("✅ Semantic Search")
    st.write("✅ Multi-Agent AI")
    st.write("✅ RAG Pipeline")
    st.write("✅ NVIDIA API Integration")

    st.markdown("---")

    st.subheader("🧠 Available Agents")

    st.write("📘 Research Agent")
    st.write("📝 Summary Agent")
    st.write("📧 Email Agent")
    st.write("💡 Insight Agent")

# MAIN TITLE
st.title("📄 AI Workflow Agent")

st.write("Upload a PDF and interact with AI agents.")

# FILE UPLOADER
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# MAIN WORKFLOW
if uploaded_file is not None:

    st.success("✅ PDF uploaded successfully!")

    # Save uploaded file
    saved_path = save_uploaded_file(uploaded_file)

    st.info(f"📁 File saved at: {saved_path}")

    # Extract text
    raw_text = extract_text_from_pdf(uploaded_file)

    # Clean text
    cleaned_text = clean_text(raw_text)

    # Create chunks
    chunks = chunk_text(cleaned_text)

    # Metrics Dashboard
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Chunks", len(chunks))

    with col2:
        st.metric("🧠 AI Agents", 4)

    with col3:
        st.metric("⚡ Workflow", "Active")

    st.markdown("---")

    # Create vector store
    vector_store = create_vector_store(chunks)

    # Save vector store
    save_vector_store(vector_store)

    st.success("✅ Vector database created successfully!")

    # Expandable Chunk Viewer
    with st.expander("📦 View Text Chunks"):

        for i, chunk in enumerate(chunks[:5]):

            st.write(f"### Chunk {i+1}")
            st.write(chunk)
            st.markdown("---")

    # Extracted Text
    st.subheader("📚 Extracted & Cleaned Text")

    st.text_area(
        "PDF Content",
        cleaned_text,
        height=300
    )

    st.markdown("---")

    # AI Agent Selector
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
        st.info("📘 Best for technical explanations and detailed analysis.")

    elif agent_type == "Summary Agent":
        st.info("📝 Best for concise summaries and simplified explanations.")

    elif agent_type == "Email Agent":
        st.info("📧 Generates professional email drafts from document context.")

    elif agent_type == "Insight Agent":
        st.info("💡 Extracts trends, insights, and observations.")

    st.markdown("---")

    # CHATBOT SECTION
    st.subheader("💬 Chat with Your PDF")

    user_question = st.text_area(
        "💬 Ask Questions About Your Document",
        placeholder="Example: Summarize this document or explain key findings...",
        height=120
    )

    # AI RESPONSE BUTTON
    if st.button("🚀 Generate AI Response"):

        if user_question:

            with st.spinner("AI is analyzing your document..."):

                # Load saved vector DB
                vector_store = load_vector_store()

                # Generate AI response
                answer = ask_question(
                    vector_store,
                    user_question,
                    agent_type
                )

            # Store conversation history
            st.session_state.chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                    "agent": agent_type
                }
            )

    # DISPLAY CHAT HISTORY
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

    st.info("📄 Upload a PDF to start the AI workflow.")

    st.markdown("""
## 🔄 Workflow Pipeline

PDF Upload → Text Extraction → Cleaning → Embeddings → FAISS Vector DB → Multi-Agent AI Chatbot
""")

# FOOTER
st.markdown("---")

st.caption(
    "Built using Streamlit, LangChain, FAISS, and OpenAI"
)