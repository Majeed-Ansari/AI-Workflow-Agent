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

st.set_page_config(page_title="AI Workflow Agent")

st.title("📄 AI Workflow Agent")
st.write("Upload a PDF and chat with it.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")

    # Save uploaded file
    saved_path = save_uploaded_file(uploaded_file)

    st.info(f"File saved at: {saved_path}")

    # Extract text
    raw_text = extract_text_from_pdf(uploaded_file)

    # Clean text
    cleaned_text = clean_text(raw_text)

    # Create chunks
    chunks = chunk_text(cleaned_text)

    st.subheader("✂️ Text Chunks")
    st.write(f"Total Chunks Created: {len(chunks)}")

    # Create vector store
    vector_store = create_vector_store(chunks)

    # Save vector store
    save_vector_store(vector_store)

    st.success("✅ Vector database created successfully!")

    # Sample chunks
    st.subheader("📦 Sample Chunks")

    for i, chunk in enumerate(chunks[:3]):

        st.write(f"### Chunk {i+1}")
        st.write(chunk)

    # Show extracted text
    st.subheader("📚 Extracted & Cleaned Text")

    st.text_area(
        "PDF Content",
        cleaned_text,
        height=400
    )

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

    st.info(f"Selected Agent: {agent_type}")

    # CHATBOT SECTION
    st.subheader("💬 Chat with Your PDF")

    user_question = st.text_input(
        "Ask a question from the document"
    )

    if user_question:

        with st.spinner("Thinking..."):

            # Load saved FAISS vector DB
            vector_store = load_vector_store()

            # Ask AI
            answer = ask_question(
                vector_store,
                user_question,
                agent_type
            )

        st.subheader("🤖 AI Response")

        st.write(answer)

else:

    st.write("Upload a PDF and extract text.")

    st.markdown("""
## 🔄 Workflow Pipeline

PDF Upload → Text Extraction → Cleaning → Embeddings → Vector DB → AI Chatbot
""")