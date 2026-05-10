import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def grade_documents(state):
    """
    Check if the retrieved documents are actually relevant
    to the question before wasting a generation call.
    """
    question = state["question"]
    docs = state["documents"]

    if not docs:
        return {"relevant": False}

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a relevance checker.

Does the context contain information useful for answering the question?
Reply with ONLY one word: YES or NO.

Question: {question}

Context:
{context}

Answer:"""

    response = llm.invoke(prompt)
    raw = response.content.strip().upper().replace("*", "").replace(".", "")
    relevant = "YES" in raw

    return {"relevant": relevant}
