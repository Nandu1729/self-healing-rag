import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def generate(state):
    question = state["question"]
    docs = state["documents"]

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
Be concise and accurate. Do not make up information.

Question: {question}

Context:
{context}

Answer:"""

    response = llm.invoke(prompt)

    return {"answer": response.content.strip()}
