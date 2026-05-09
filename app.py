import streamlit as st
from src.pdf_processor import (
    extract_text_from_pdf,
    clean_text,
    save_uploaded_file
)

st.set_page_config(page_title="AI Workflow Agent")

st.title("📄 AI Workflow Agent")
st.write("Upload a PDF and extract text.")

# File uploader
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    st.success("PDF uploaded successfully!")
    saved_path = save_uploaded_file(uploaded_file)

    st.info(f"File saved at: {saved_path}")

    # Extract text
    raw_text = extract_text_from_pdf(uploaded_file)

    # Clean text
    cleaned_text = clean_text(raw_text)

    st.subheader("📚 Extracted & Cleaned Text")

    # Display text
    st.text_area(
        "PDF Content",
        cleaned_text,
        height=400
    )
    st.write("Upload a PDF and extract text.")
    st.markdown("""
## 🔄 Workflow Pipeline

PDF Upload → Text Extraction → Cleaning → AI Processing
""")