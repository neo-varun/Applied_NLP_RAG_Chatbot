from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})


def search_reviews(question):
    docs = retriever.get_relevant_documents(question)
    return docs
