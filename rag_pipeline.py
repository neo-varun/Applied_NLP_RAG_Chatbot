import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from vector_database import create_vector_db


def load_documents():
    df = pd.read_csv("data/amazon_review.csv")

    df = df[["asin", "reviewText"]].dropna()

    docs = []
    for _, row in df.iterrows():
        text = f"Product: {row['asin']}\nReview: {row['reviewText']}"
        docs.append(Document(page_content=text, metadata={"asin": row["asin"]}))

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    return splitter.split_documents(docs)


docs = load_documents()
chunks = chunk_documents(docs)
create_vector_db(chunks)
