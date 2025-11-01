# JosefGPT Local

Local-first knowledge retrieval and coaching assistant built around hybrid retrieval with OpenAI reasoning.

## Features
- Chunk and embed PDFs, EPUBs, Markdown, and text files from `books/`, `texts/`, or `data/` directories.
- Streamlit UI with chat history, configurable retrieval/generation settings, and source previews.
- Typer-based CLI for ingestion, terminal chat, and launching the UI.
- Flexible configuration through `.env` without touching code, including an offline heuristic fallback when an OpenAI key is missing.

## Quickstart
1. **Install dependencies**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Configure environment**
   - Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
3. **Add knowledge sources**
   - Drop PDFs, EPUBs, Markdown, or text files into `books/`, `texts/`, or `data/`.
4. **Ingest embeddings**
   ```bash
   python -m app.cli ingest
   ```
5. **Launch the UI**
   ```bash
   streamlit run app/ui.py
   ```

## CLI Commands
Run `python -m app.cli --help` for the full menu.

- `python -m app.cli ingest`  
  Rebuilds the Chroma database. Use `-s path/to/dir` to ingest custom folders.

- `python -m app.cli chat`  
  Starts an interactive terminal chat. Flags such as `--top-k`, `--temperature`, and `--max-tokens` override defaults, and `--hide-sources` suppresses source summaries.

- `python -m app.cli serve`  
  Convenience wrapper around `streamlit run app/ui.py`.

## Streamlit UI
- Adjust retrieval/generation parameters from the sidebar; settings persist during the session.
- Every reply lists supporting source excerpts with similarity scores and chunk identifiers for quick verification.

## Configuration
Environment variables (see `.env.example`):

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENAI_MODEL` | Chat completion model. | `gpt-5-turbo` |
| `USE_OPENAI_EMBEDDINGS` | Switch between OpenAI and local SentenceTransformer embeddings. | `false` |
| `EMBEDDING_MODEL` | OpenAI embedding model name. | `text-embedding-3-large` |
| `TOP_K` | Default retrieved chunks per query. | `6` |
| `MAX_TOKENS` | Default completion max tokens. | `900` |
| `TEMPERATURE` | Default completion temperature. | `0.3` |
| `EMBEDDINGS_PATH` | Location for the ChromaDB store. | `embeddings/` |
| `SOURCE_DIRS` | Comma-separated list of directories to scan. | `books,texts,data` |
| `LLM_MODE` | `openai`, `offline`, or `auto` (fallback to offline when no key). | `auto` |

## Development & Testing
- Format/compile checks:
  ```bash
  python3 -m py_compile ingest_books.py app/*.py
  ```
- Run the automated test suite (once added in `tests/`):
  ```bash
  pytest
  ```
- Clean embeddings/data quickly by removing the `embeddings/` directory (listed in `.gitignore`).

## Project Layout
```

## Offline Mode
- Set `LLM_MODE=offline` (or omit an `OPENAI_API_KEY`) to run without external API calls.
- In offline mode, JosefGPT synthesises heuristic guidance from retrieved context; UI and CLI clearly flag this mode.
.
├── app/
│   ├── cli.py          # Typer CLI
│   ├── config.py       # Central configuration
│   ├── query_engine.py # Retrieval + LLM orchestration
│   └── ui.py           # Streamlit interface
├── ingest_books.py     # Ingestion pipeline (shared with CLI)
├── requirements.txt
└── README.md
```
