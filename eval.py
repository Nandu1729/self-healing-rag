import os
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, api_key=os.getenv("GROQ_API_KEY"))
judge_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=os.getenv("GROQ_API_KEY"))

SAMPLE_DOCS = [
    Document(page_content="Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve performance without being explicitly programmed. It uses algorithms to parse data and make decisions."),
    Document(page_content="RAG (Retrieval-Augmented Generation) combines information retrieval with language model generation. It retrieves relevant documents from a knowledge base and uses them as context for generating accurate, grounded responses."),
    Document(page_content="LangGraph is a library for building stateful, multi-actor applications with language models. It extends LangChain with cyclical graph support, enabling self-correction and multi-step reasoning workflows."),
    Document(page_content="Hallucination in AI refers to when a language model generates information that is factually incorrect or not grounded in the provided context. It is a major challenge in deploying LLMs in production systems."),
    Document(page_content="Groq is an AI infrastructure company that provides high-speed LLM inference. Their hardware accelerates transformer model execution significantly compared to traditional GPU-based solutions."),
]

CI_DATASET = [
    {"question": "What is machine learning?", "expected": "Machine learning is a subset of artificial intelligence that enables systems to learn from data and improve performance without being explicitly programmed."},
    {"question": "What is RAG?", "expected": "RAG is Retrieval-Augmented Generation, combining information retrieval with language model generation to produce accurate, grounded responses."},
    {"question": "What is LangGraph?", "expected": "LangGraph is a library for building stateful, multi-actor applications with language models, extending LangChain with cyclical graph support."},
    {"question": "What is hallucination in AI?", "expected": "Hallucination in AI is when a language model generates factually incorrect or ungrounded information, a major challenge in production LLM systems."},
    {"question": "What is Groq?", "expected": "Groq is an AI infrastructure company providing high-speed LLM inference through specialized hardware acceleration."},
]

W = 58


def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=SAMPLE_DOCS, embedding=embeddings)


def call_rag(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=2)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""Answer the question using ONLY the context below. Be concise.

Question: {question}

Context:
{context}

Answer:"""
    start = time.time()
    response = llm.invoke(prompt)
    latency_ms = (time.time() - start) * 1000
    answer = response.content.strip()
    cost = (len(prompt.split()) * 1.3 * 0.05 + len(answer.split()) * 1.3 * 0.08) / 1_000_000
    return answer, latency_ms, cost


def _retry_invoke(prompt):
    for attempt in range(5):
        try:
            return judge_llm.invoke(prompt).content.strip().upper()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 15 * (attempt + 1)
                print(f"    [rate limit] waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return ""


def judge_all(question, answer, expected):
    prompt = f"""Evaluate the Given Answer against the Expected Answer for the question.

Question: {question}
Expected Answer: {expected}
Given Answer: {answer}

Answer all 3:
1. HALLUCINATION: Does the Given Answer contain facts NOT in the Expected Answer or clearly wrong?
2. RELEVANCY: Is the Given Answer relevant and on-topic?
3. FAITHFULNESS: Does the Given Answer faithfully cover the main points of the Expected Answer?

Reply in exactly this format:
HALLUCINATION: YES or NO
RELEVANCY: YES or NO
FAITHFULNESS: YES or NO"""

    raw = _retry_invoke(prompt)

    def extract(label):
        for line in raw.splitlines():
            if label in line:
                return "YES" in line
        return False

    return {
        "hallucinated": extract("HALLUCINATION"),
        "relevant": extract("RELEVANCY"),
        "faithful": extract("FAITHFULNESS"),
    }


def compute_stats(results):
    n = len(results)
    latencies = sorted([r["latency_ms"] for r in results])
    return {
        "total": n,
        "hallucination_rate": sum(1 for r in results if r["hallucinated"]) / n,
        "relevancy_score": sum(1 for r in results if r["relevant"]) / n,
        "faithfulness_score": sum(1 for r in results if r["faithful"]) / n,
        "latency_p50": latencies[n // 2],
        "latency_p95": latencies[min(int(n * 0.95), n - 1)],
        "total_cost": sum(r["cost"] for r in results),
        "avg_cost": sum(r["cost"] for r in results) / n,
    }


def print_dashboard(stats, gate):
    print("\n" + "═" * W)
    print("   Self-Healing RAG — Eval Dashboard")
    print("═" * W)
    print(f"\n  Samples evaluated : {stats['total']}")
    print(f"\n  {'METRIC':<32} {'VALUE':<12} STATUS")
    print("  " + "─" * (W - 2))

    def row(label, value, ok):
        print(f"  {label:<32} {value:<12} {'PASS' if ok else 'FAIL'}")

    row("Hallucination Rate", f"{stats['hallucination_rate']*100:.1f}%", not gate["hallucination_failed"])
    row("Answer Relevancy", f"{stats['relevancy_score']*100:.1f}%", not gate["relevancy_failed"])
    row("Faithfulness", f"{stats['faithfulness_score']*100:.1f}%", not gate["faithfulness_failed"])
    row("Latency p50", f"{stats['latency_p50']:.0f} ms", True)
    row("Latency p95", f"{stats['latency_p95']:.0f} ms", not gate["latency_failed"])
    row("Avg Cost / Query", f"${stats['avg_cost']:.6f}", True)
    row("Total Cost", f"${stats['total_cost']:.4f}", True)

    print("\n" + "─" * W)
    if gate["passed"]:
        print("\n  GATE: PASSED — safe to merge")
    else:
        print("\n  GATE: FAILED — merge blocked")
        if gate["hallucination_failed"]: print("    ✗ Hallucination rate too high")
        if gate["relevancy_failed"]:     print("    ✗ Relevancy score too low")
        if gate["faithfulness_failed"]:  print("    ✗ Faithfulness score too low")
        if gate["latency_failed"]:       print("    ✗ p95 latency exceeded SLA")
    print("═" * W + "\n")


def evaluate_gate(stats):
    return {
        "passed": stats["hallucination_rate"] <= 0.10 and stats["relevancy_score"] >= 0.75 and stats["faithfulness_score"] >= 0.70 and stats["latency_p95"] <= 5000,
        "hallucination_failed": stats["hallucination_rate"] > 0.10,
        "relevancy_failed": stats["relevancy_score"] < 0.75,
        "faithfulness_failed": stats["faithfulness_score"] < 0.70,
        "latency_failed": stats["latency_p95"] > 5000,
    }


def main():
    import sys
    print("\n" + "═" * W)
    print("   Self-Healing RAG — CI/CD Eval Pipeline")
    print("   Powered by Groq + LangGraph")
    print("═" * W)

    print("\n  Building in-memory vector store...")
    vectorstore = build_vectorstore()

    results = []
    total = len(CI_DATASET)
    print(f"  Running eval on {total} samples...\n")

    for i, item in enumerate(CI_DATASET, 1):
        q, exp = item["question"], item["expected"]
        print(f"  [{i:02d}/{total}] {q[:55]}...")
        answer, latency_ms, cost = call_rag(q, vectorstore)
        scores = judge_all(q, answer, exp)
        results.append({**scores, "latency_ms": latency_ms, "cost": cost})

    stats = compute_stats(results)
    gate = evaluate_gate(stats)
    print_dashboard(stats, gate)

    if not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
