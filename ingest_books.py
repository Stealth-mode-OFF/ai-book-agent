from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ebooklib
import fitz
from ebooklib import epub
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.config import get_settings

settings = get_settings()
USE_OPENAI_EMBEDDINGS = settings.use_openai_embeddings
EMBEDDING_MODEL = settings.embedding_model
SOURCE_DIRS = settings.source_dirs
SUPPORTED_SUFFIXES = {suffix.lower() for suffix in settings.supported_suffixes}
TEXT_SUFFIXES = {suffix.lower() for suffix in settings.text_suffixes}

_local_encoder: Optional[SentenceTransformer] = None
_openai_client: Optional[OpenAI] = None


def iter_source_files(directories: Iterable[Path]) -> Iterable[Tuple[Path, Path]]:
    for base_dir in directories:
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not path.name.startswith(".")
            ):
                yield path, base_dir


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            with fitz.open(path) as doc:
                return "\n".join(page.get_text("text") for page in doc)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️ {path.name}: PDF parsing failed ({exc}).")
            return ""
    if suffix == ".epub":
        try:
            book = epub.read_epub(path)
            return "\n".join(
                item.get_body_content().decode("utf-8", errors="ignore")
                for item in book.get_items()
                if item.get_type() == ebooklib.ITEM_DOCUMENT
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️ {path.name}: EPUB parsing failed ({exc}).")
            return ""
    if suffix in TEXT_SUFFIXES:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def get_local_encoder() -> SentenceTransformer:
    global _local_encoder
    if _local_encoder is None:
        _local_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_encoder


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    if USE_OPENAI_EMBEDDINGS:
        client = get_openai_client()
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunks)
        return [d.embedding for d in resp.data]
    encoder = get_local_encoder()
    return encoder.encode(chunks, show_progress_bar=False).tolist()


def source_key(path: Path, base_dir: Path) -> str:
    return f"{base_dir.name}/{path.relative_to(base_dir).as_posix()}"


def ingest_file(path: Path, base_dir: Path, collection, splitter) -> int:
    text = extract_text(path)
    if not text.strip():
        print(f"⚠️ {source_key(path, base_dir)}: no readable text (maybe scan/OCR needed).")
        return 0
    chunks = splitter.split_text(text)
    if not chunks:
        print(f"⚠️ {source_key(path, base_dir)}: splitter produced no chunks.")
        return 0
    embeddings = embed_chunks(chunks)
    src = source_key(path, base_dir)
    collection.delete(where={"source": src})
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": src, "chunk": idx} for idx in range(len(chunks))],
        ids=[f"{src}#{idx}" for idx in range(len(chunks))],
    )
    return len(chunks)


def ingest_all(source_dirs: Optional[Iterable[Path]] = None):
    directories = list(source_dirs or SOURCE_DIRS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    client = chromadb.PersistentClient(path=settings.embeddings_path.as_posix())
    collection = client.get_or_create_collection("josef_knowledge")
    files = list(iter_source_files(directories))
    if not files:
        print("⚠️ No sources found. Add files into 'books/', 'texts/' or 'data/'.")
        return {"files": 0, "chunks": 0, "scanned": 0}

    total_files = 0
    total_chunks = 0
    for path, base_dir in tqdm(files, desc="Ingesting", unit="file"):
        try:
            count = ingest_file(path, base_dir, collection, splitter)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"⚠️ {source_key(path, base_dir)}: ingestion failed ({exc}).")
            continue
        if count:
            total_files += 1
            total_chunks += count
            print(f"✅ {source_key(path, base_dir)}: stored {count} chunks.")

    print(f"🏁 Done. {total_chunks} chunks saved from {total_files} files.")
    return {"files": total_files, "chunks": total_chunks, "scanned": len(files)}


def main():
    ingest_all()


if __name__ == "__main__":
    main()
