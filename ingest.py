"""Document loading (PDF + CSV), chunking, embedding and ChromaDB storage.
Run:  python ingest.py
"""
import os
import re
import random
from pathlib import Path
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (DATA_DIR, CHROMA_DIR, COLLECTION, EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)


def load_pdf(path: Path):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            m = re.search(r"^(Section [^\n]+)", text, re.M)
            docs.append(Document(page_content=text, metadata={
                "source": path.name, "page": i,
                "section": m.group(1) if m else "General", "doc_type": "policy"}))
    return docs


def load_csv(path: Path):
    df = pd.read_csv(path)
    return [Document(page_content=f"Q: {r.query} | Resolution: {r.resolution}",
                     metadata={"source": path.name, "page": int(i) + 1,
                               "section": str(r.category), "doc_type": "ticket"})
            for i, r in df.iterrows()]


def load_all(data_dir: Path = DATA_DIR):
    docs = []
    for p in sorted(data_dir.iterdir()):
        if p.suffix.lower() == ".pdf":
            docs += load_pdf(p)
        elif p.suffix.lower() == ".csv":
            docs += load_csv(p)
    return docs


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)  # metadata is preserved on every chunk


def build_index():
    """(Re)build the persistent ChromaDB collection. Returns number of chunks."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    chunks = chunk(load_all())
    if not chunks:
        raise RuntimeError("No documents found in /data.")
    emb = OpenAIEmbeddings(model=EMBED_MODEL)
    try:
        Chroma(collection_name=COLLECTION, embedding_function=emb,
               persist_directory=str(CHROMA_DIR)).delete_collection()
    except Exception:
        pass
    Chroma.from_documents(chunks, emb, collection_name=COLLECTION,
                          persist_directory=str(CHROMA_DIR),
                          collection_metadata={"hnsw:space": "cosine"})
    return len(chunks)


def inspect_random_chunks(n=10):
    """Step-3 verification: print n random chunks with their metadata."""
    chunks = chunk(load_all())
    for c in random.sample(chunks, min(n, len(chunks))):
        print(c.metadata, "\n", c.page_content[:200].replace("\n", " "), "\n" + "-" * 60)


if __name__ == "__main__":
    print("Chunks indexed:", build_index())
