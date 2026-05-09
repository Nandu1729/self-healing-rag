from app.graph import app

print("Self-Healing RAG Started!")
print("Type 'exit' to quit.")

while True:

    question = input("\nAsk Question: ")

    if question.lower() == "exit":
        print("Exiting...")
        break

    result = app.invoke({
        "question": question,
        "retry_count": 0
    })

    print("\nFinal Answer:")
    print(result["answer"])

    print("\n" + "="*50)