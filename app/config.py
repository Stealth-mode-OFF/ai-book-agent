from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional
import os

from dotenv import load_dotenv

load_dotenv()


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _split_paths(value: str | None, default: Iterable[str]) -> List[Path]:
    if not value:
        return [Path(item).expanduser() for item in default]
    parts = [item.strip() for item in value.split(",")]
    return [Path(item).expanduser() for item in parts if item]


@dataclass(frozen=True)
class Settings:
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-turbo")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    llm_mode: str = os.getenv("LLM_MODE", "auto")
    top_k: int = _int(os.getenv("TOP_K"), 6)
    max_tokens: int = _int(os.getenv("MAX_TOKENS"), 900)
    temperature: float = _float(os.getenv("TEMPERATURE"), 0.3)
    use_openai_embeddings: bool = _bool(os.getenv("USE_OPENAI_EMBEDDINGS"), False)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    embeddings_path: Path = Path(os.getenv("EMBEDDINGS_PATH", "embeddings"))
    source_dirs: List[Path] = field(
        default_factory=lambda: _split_paths(
            os.getenv("SOURCE_DIRS"),
            default=("books", "texts", "data"),
        )
    )
    supported_suffixes: List[str] = field(
        default_factory=lambda: [".pdf", ".epub", ".txt", ".md", ".mdx"]
    )
    text_suffixes: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".mdx"]
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
