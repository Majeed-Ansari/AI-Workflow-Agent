from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.docstore.document import Document


def chunk_text(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    return chunks


def create_vector_store(chunks):

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents = []

    current_source = "Unknown"

    for chunk in chunks:

        if "DOCUMENT:" in chunk:

            lines = chunk.split("\n")

            for line in lines:

                if "DOCUMENT:" in line:
                    current_source = line.replace(
                        "DOCUMENT:",
                        ""
                    ).strip()

        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": current_source
                }
            )
        )

    vector_store = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vector_store


def save_vector_store(vector_store):

    vector_store.save_local("faiss_index")
    
def load_vector_store():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store