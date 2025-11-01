from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional

import typer

from app.config import get_settings
from app.llm import get_chat_llm
from app.query_engine import answer_with_context
from ingest_books import SOURCE_DIRS, ingest_all

cli = typer.Typer(help="JosefGPT Local command line interface.")


@cli.command()
def ingest(
    source_dir: list[Path] = typer.Option(
        None,
        "--source-dir",
        "-s",
        help="Override source directories (can be passed multiple times).",
    )
):
    """Ingest knowledge sources into the local ChromaDB store."""
    directories: Iterable[Path] = source_dir or SOURCE_DIRS
    typer.echo("📥 Starting ingestion...")
    result = ingest_all(directories)
    typer.echo(
        f"🏁 Done. {result['chunks']} chunks saved from {result['files']} files "
        f"(scanned {result['scanned']} potential files)."
    )


@cli.command()
def chat(
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        "-k",
        help="Override number of retrieved chunks per question.",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Override completion temperature.",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        "-m",
        help="Override maximum completion tokens.",
    ),
    show_sources: bool = typer.Option(
        True,
        "--show-sources/--hide-sources",
        help="Toggle printing supporting source summaries.",
    ),
):
    """Interactive CLI chat that mirrors the Streamlit experience."""
    settings = get_settings()
    llm = get_chat_llm()
    typer.echo("🧠 JosefGPT (type 'exit' or Ctrl+C to quit)")
    typer.echo(
        f"Defaults — k={settings.top_k}, temp={settings.temperature}, "
        f"max_tokens={settings.max_tokens}"
    )
    typer.echo(f"LLM mode: {llm.mode} ({llm.model_name})")
    if llm.mode == "offline":
        typer.echo("⚠️ Offline mode active — responses use local heuristics.")
    if top_k is not None or temperature is not None or max_tokens is not None:
        typer.echo(
            "Overrides applied — "
            + ", ".join(
                bit
                for bit in [
                    f"k={top_k}" if top_k is not None else "",
                    f"temp={temperature}" if temperature is not None else "",
                    f"max_tokens={max_tokens}" if max_tokens is not None else "",
                ]
                if bit
            )
        )
    try:
        while True:
            question = typer.prompt("🧍‍♂️ You").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break
            typer.echo("🤖 JosefGPT is thinking...")
            result = answer_with_context(
                question,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            typer.echo(f"🤖 JosefGPT: {result['answer']}\n")
            if show_sources:
                sources = result.get("sources") or []
                if sources:
                    typer.echo("📚 Sources:")
                    for idx, source in enumerate(sources, start=1):
                        chunk = source.get("chunk")
                        score = source.get("score")
                        details = []
                        if chunk is not None:
                            details.append(f"chunk {chunk}")
                        if isinstance(score, (int, float)):
                            details.append(f"score {score:.2f}")
                        suffix = f" ({', '.join(details)})" if details else ""
                        typer.echo(f"  {idx}. {source.get('source', 'Unknown')}{suffix}")
                        preview = source.get("preview")
                        if preview:
                            typer.echo(f"     {preview}")
            llm_result = result.get("llm")
            if llm_result:
                typer.echo(
                    f"[mode: {llm_result.get('mode')} | model: {llm_result.get('model')}]"
                )
            typer.echo("")
    except (EOFError, KeyboardInterrupt):
        typer.echo("\n👋 Goodbye!")


@cli.command()
def serve():
    """Launch the Streamlit web app."""
    typer.echo("🚀 Launching Streamlit UI at http://localhost:8501 ...")
    subprocess.run(["streamlit", "run", "app/ui.py"], check=True)


if __name__ == "__main__":
    cli()
