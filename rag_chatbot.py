import os
from google import genai
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(persist_directory="vector_db", embedding_function=embedding_model)

retriever = vectordb.as_retriever(search_kwargs={"k": 5})


def rag_chatbot(question):

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are a product reviewer for a MicroSD card.
Answer only using the provided context.
If answer not found, say 'Not available in reviews'.

context:
{context}

question:
{question}
"""
    response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)

    return response.text
