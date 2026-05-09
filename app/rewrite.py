from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")


def rewrite(state):
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    prompt = f'''
Rewrite this question to improve
document retrieval quality.

Original Question:
{question}

Rewritten Question:
'''

    response = llm.invoke(prompt)
    rewritten_question = response.content.strip()

    print(f"\nRewritten Question (attempt {retry_count + 1}):")
    print(rewritten_question)

    return {
        "question": rewritten_question,
        "retry_count": retry_count + 1,
    }
