import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


def critic(state):
    question = state["question"]
    docs = state["documents"]
    answer = state["answer"]

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a strict fact-checker.

Decide if the answer is fully supported by the context.
If the answer contains any information NOT in the context, reply FAIL.
If the answer is fully grounded in the context, reply PASS.

Question: {question}

Context:
{context}

Answer:
{answer}

Reply with ONLY one word — PASS or FAIL. Nothing else."""

    response = llm.invoke(prompt)

    raw = response.content.strip().upper().replace("*", "").replace(".", "")
    grade = "PASS" if "PASS" in raw else "FAIL"

    return {"grade": grade}
