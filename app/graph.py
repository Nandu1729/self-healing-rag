from langgraph.graph import StateGraph, END

from app.state import GraphState
from app.retrieve import retrieve
from app.generate import generate
from app.critic import critic
from app.rewrite import rewrite

MAX_RETRIES = 3


def route(state):
    grade = state["grade"]
    retry_count = state.get("retry_count", 0)

    print(f"\nCritic Decision: {grade} (attempt {retry_count + 1}/{MAX_RETRIES + 1})")

    if grade == "PASS":
        return END

    if retry_count >= MAX_RETRIES:
        print("Max retries reached — returning graceful fallback.")
        return "fallback"

    return "rewrite"


def fallback(state):
    return {
        "answer": (
            "I don't have enough information in the provided documents "
            "to answer this question confidently. Please try rephrasing "
            "your question or provide additional context."
        )
    }


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("critic", critic)
workflow.add_node("rewrite", rewrite)
workflow.add_node("fallback", fallback)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critic")
workflow.add_edge("fallback", END)

workflow.add_conditional_edges(
    "critic",
    route,
    {
        END: END,
        "rewrite": "rewrite",
        "fallback": "fallback",
    }
)

workflow.add_edge("rewrite", "retrieve")

app = workflow.compile()
