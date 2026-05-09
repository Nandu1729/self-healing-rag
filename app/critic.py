from langchain_ollama import ChatOllama

llm = ChatOllama(model="phi3")


def critic(state):

    question = state["question"]
    docs = state["documents"]
    answer = state["answer"]

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    prompt = f'''
You are a strict evaluator.

Check whether the answer is supported
by the provided context.

Question:
{question}

Context:
{context}

Answer:
{answer}

IMPORTANT:
Return ONLY one word.

PASS
or
FAIL

Do not explain.
Do not add markdown.
'''

    response = llm.invoke(prompt)

    grade = response.content.strip().upper()

    # Remove markdown symbols if model adds them
    grade = grade.replace("*", "")

    return {
        "grade": grade
    }