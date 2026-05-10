import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def rewrite(state):
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    prompt = f"""Rewrite the question below to improve search and document retrieval.
Make it more specific and use different keywords.
Return ONLY the rewritten question — no explanation, no quotes.

Original: {question}

Rewritten:"""

    response = llm.invoke(prompt)
    rewritten = response.content.strip().strip('"').strip("'")

    return {
        "question": rewritten,
        "retry_count": retry_count + 1,
    }
