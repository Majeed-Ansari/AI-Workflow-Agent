import os
from pypdf import PdfReader
import re


def extract_text_from_pdf(pdf_file):
    """
    Extract text from uploaded PDF
    """
    pdf_reader = PdfReader(pdf_file)

    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted + "\n"

    return text


def clean_text(text):
    """
    Clean extracted text
    """

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove unwanted characters
    text = re.sub(r"[^\w\s.,!?;:()\-]", "", text)

    return text.strip()

def save_uploaded_file(uploaded_file):

    upload_folder = "uploads"

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path