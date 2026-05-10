"""
Ingest documents into ChromaDB.

Usage:
    python app/ingest.py                  # loads all PDFs from data/
    python app/ingest.py data/myfile.pdf  # loads a specific file
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SKIP_SECTIONS = ["REFERENCES", "BIBLIOGRAPHY"]


def load_docs(path: str) -> list:
    if os.path.isfile(path):
        print(f"Loading: {path}")
        loader = PyPDFLoader(path)
        return loader.load()

    print(f"Loading all PDFs from: {path}/")
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def clean(docs: list) -> list:
    cleaned = []
    for doc in docs:
        if any(sec in doc.page_content for sec in SKIP_SECTIONS):
            continue
        if len(doc.page_content.strip()) < 50:
            continue
        cleaned.append(doc)
    return cleaned


def ingest(path: str = DATA_DIR):
    print("\n Self-Healing RAG — Document Ingestion")
    print("─" * 40)

    docs = load_docs(path)
    docs = clean(docs)
    print(f"Pages loaded : {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")

    print("Generating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"Stored in    : {CHROMA_DIR}/")
    print("Done!\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    ingest(target)
