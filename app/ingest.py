from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import os

# Load PDF
loader = PyPDFLoader("data/B19_RP.pdf")
docs = loader.load()
cleaned_docs = []

for doc in docs:

    text = doc.page_content

    # Skip references-heavy pages
    if "REFERENCES" in text:
        continue

    cleaned_docs.append(doc)

docs = cleaned_docs

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Documents embedded successfully!")