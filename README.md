# Self-Healing RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** system that critiques its own output and retries — instead of hallucinating, it self-corrects.

Built with **LangGraph** as a stateful, cyclical workflow (not a simple linear chain).

---

## How It Works

```
User Question
      │
      ▼
  [Retrieve] ──── vector store (ChromaDB + MMR)
      │
      ▼
  [Generate] ──── local LLM (Ollama llama3)
      │
      ▼
   [Critic] ──── strict grounding check (Ollama phi3)
      │
   ┌──┴──┐
  PASS  FAIL
   │      │
   ▼      ▼
 Answer  [Rewrite] ──── reformulate query
           │
           └──▶ [Retrieve] (retry loop, max 3)
                    │
               (if still FAIL)
                    ▼
              "I don't have enough
               information..." ✓
```

### Pipeline Nodes

| Node | Role |
|------|------|
| **Retrieve** | Fetches top-k document chunks from ChromaDB using MMR (max marginal relevance) |
| **Generate** | Produces an answer using only the retrieved context |
| **Critic** | Evaluates whether the answer is actually grounded in the context or hallucinated |
| **Rewrite** | Reformulates the query to improve retrieval on the next attempt |
| **Fallback** | Returns a graceful "I don't have enough information" after max retries |

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** — stateful, cyclical multi-agent workflow
- **[ChromaDB](https://www.trychroma.com/)** — local vector store with persistent storage
- **[Ollama](https://ollama.com/)** — local LLMs (llama3 for generation, phi3 for critic)
- **[sentence-transformers](https://www.sbert.net/)** — `all-MiniLM-L6-v2` for embeddings
- **[LangChain](https://langchain.com/)** — document loaders, text splitters, retriever abstraction

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

```bash
# Pull the required models
ollama pull llama3
ollama pull phi3
```

### Install

```bash
git clone https://github.com/yourusername/self-healing-rag.git
cd self-healing-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Ingest Documents

Place your PDF(s) in the `data/` folder, then update the path in `app/ingest.py` and run:

```bash
python app/ingest.py
```

This chunks your documents and stores embeddings in `chroma_db/`.

### Run

```bash
python main.py
```

```
Self-Healing RAG Started!
Type 'exit' to quit.

Ask Question: What are the main findings?

Critic Decision: FAIL (attempt 1/4)

Rewritten Question (attempt 1):
What are the key conclusions and findings presented in the document?

Critic Decision: PASS (attempt 2/4)

Final Answer:
...
```

---

## Key Design Decisions

**Why LangGraph over a simple chain?**  
LangGraph supports cycles — the retry loop (rewrite → retrieve → generate → critic) cannot be expressed in a linear LangChain chain.

**Why a separate critic model (phi3)?**  
Smaller, faster, and less likely to rationalise a bad answer. The critic's only job is binary grounding evaluation.

**Why MMR retrieval?**  
Maximum Marginal Relevance reduces redundancy in retrieved chunks, giving the generator more diverse context to work with.

**Why a graceful fallback instead of infinite retries?**  
After 3 failed attempts, returning "I don't have enough information" is more honest and useful than hallucinating a confident-sounding wrong answer.

---

## Project Structure

```
self-healing-rag/
├── app/
│   ├── state.py       # LangGraph state schema
│   ├── graph.py       # Workflow definition + routing logic
│   ├── retrieve.py    # ChromaDB retriever
│   ├── generate.py    # LLM answer generation
│   ├── critic.py      # Grounding evaluation
│   ├── rewrite.py     # Query reformulation + retry counter
│   └── ingest.py      # Document ingestion pipeline
├── main.py            # CLI entry point
├── requirements.txt
└── .gitignore
```

---

## License

MIT
