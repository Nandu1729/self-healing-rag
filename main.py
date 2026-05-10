from dotenv import load_dotenv
load_dotenv()

from app.graph import app, MAX_RETRIES

DIVIDER = "─" * 50


def print_header():
    print("\n" + "═" * 50)
    print("   Self-Healing RAG Pipeline")
    print("   Powered by Groq + LangGraph")
    print("═" * 50)
    print('Type "exit" to quit.\n')


def run_query(question: str) -> dict:
    return app.invoke({
        "question": question,
        "documents": [],
        "answer": "",
        "grade": "",
        "retry_count": 0,
    })


def display_result(result: dict, original_question: str):
    final_question = result.get("question", original_question)
    grade = result.get("grade", "")
    retries = result.get("retry_count", 0)

    print(f"\n{DIVIDER}")

    if retries > 0:
        print(f"  Rewrites      : {retries}")
        print(f"  Final query   : {final_question}")

    verdict = "PASS" if grade == "PASS" else "FALLBACK" if retries >= MAX_RETRIES else grade
    print(f"  Critic verdict: {verdict}")
    print(DIVIDER)
    print("\nAnswer:\n")
    print(result["answer"])
    print(f"\n{'═' * 50}\n")


def main():
    print_header()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        print(f"\n  Retrieving and thinking...\n")

        result = run_query(question)
        display_result(result, question)


if __name__ == "__main__":
    main()
