from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def create_vector_db(chunks):
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embedding_model, persist_directory="vector_db"
    )

    vectordb.persist()
    print("Vector DB created")
