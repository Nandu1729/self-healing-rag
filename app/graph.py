from langgraph.graph import StateGraph, END
from app.state import GraphState
from app.retrieve import retrieve
from app.grader import grade_documents
from app.generate import generate
from app.critic import critic
from app.rewrite import rewrite

MAX_RETRIES = 3


def fallback(state):
    return {
        "answer": "I don't have enough information in the documents to answer this question. Please ask something related to the document."
    }


def route_after_grading(state):
    """After retrieval + relevance check: go generate or fallback immediately."""
    if state.get("relevant", False):
        return "generate"
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        return "fallback"
    return "rewrite"


def route_after_critic(state):
    """After generation: pass or retry."""
    grade = state.get("grade", "FAIL")
    retry_count = state.get("retry_count", 0)

    if grade == "PASS":
        return END

    if retry_count >= MAX_RETRIES:
        return "fallback"

    return "rewrite"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("critic", critic)
workflow.add_node("rewrite", rewrite)
workflow.add_node("fallback", fallback)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("generate", "critic")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("fallback", END)

workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"}
)

workflow.add_conditional_edges(
    "critic",
    route_after_critic,
    {END: END, "rewrite": "rewrite", "fallback": "fallback"}
)

app = workflow.compile()
