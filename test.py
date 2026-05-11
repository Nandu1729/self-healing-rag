from dotenv import load_dotenv
load_dotenv()

from app.graph import app

PASS_COLOR = "\033[92m"
FAIL_COLOR = "\033[91m"
RESET = "\033[0m"

tests = [
    {
        "name": "Pipeline returns an answer",
        "question": "What is machine learning?",
    },
    {
        "name": "Pipeline handles unknown topic gracefully",
        "question": "What is the capital of France?",
    },
]


def run_test(test):
    result = app.invoke({
        "question": test["question"],
        "documents": [],
        "answer": "",
        "grade": "",
        "retry_count": 0,
    })

    answer = result.get("answer", "").strip()
    passed = len(answer) > 0
    status = f"{PASS_COLOR}PASS{RESET}" if passed else f"{FAIL_COLOR}FAIL{RESET}"
    print(f"  [{status}] {test['name']}")
    if not passed:
        print(f"         No answer returned.")
    return passed


def main():
    print("\n══════════════════════════════════════")
    print("   Self-Healing RAG — CI Tests")
    print("══════════════════════════════════════\n")

    results = [run_test(t) for t in tests]

    total = len(results)
    passed = sum(results)
    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        print("  FAILED\n")
        exit(1)

    print("  ALL TESTS PASSED\n")


if __name__ == "__main__":
    main()
