from pathlib import Path
import os
import fitz, ebooklib
from ebooklib import epub
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

load_dotenv()
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")


def extract_text(path):
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as doc:
            return "\n".join(page.get_text("text") for page in doc)
    elif path.suffix.lower() == ".epub":
        book = epub.read_epub(path)
        return "\n".join(
            item.get_body_content().decode("utf-8", errors="ignore")
            for item in book.get_items()
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        )
    else:
        return ""


def embed_chunks(chunks):
    if USE_OPENAI_EMBEDDINGS:
        client = OpenAI()
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
        return [d.embedding for d in resp.data]
    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(chunks, show_progress_bar=True).tolist()


def main():
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    client = chromadb.PersistentClient(path="embeddings")
    collection = client.get_or_create_collection("josef_knowledge")

    for file in Path("books").glob("*"):
        text = extract_text(file)
        if not text.strip():
            print(f"⚠️ {file.name}: žádný text (možná sken/OCR?).")
            continue
        chunks = splitter.split_text(text)
        embs = embed_chunks(chunks)
        collection.add(
            documents=chunks,
            embeddings=embs,
            metadatas=[{"source": file.name}] * len(chunks),
            ids=[f"{file.name}-{i}" for i in range(len(chunks))]
        )
        print(f"✅ {file.name}: {len(chunks)} chunks uloženo.")


if __name__ == "__main__":
    main()
