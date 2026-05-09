from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")


def generate(state):

    question = state["question"]
    docs = state["documents"]

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    prompt = f'''
You are a helpful AI assistant.

Use ONLY the provided context.

Question:
{question}

Context:
{context}

Answer:
'''

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }