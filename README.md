# AI Book Agent

This repository contains a simple FastAPI web service that exposes endpoints for asking questions to a (future) AI-powered book agent.

## Overview

The service exposes two HTTP endpoints:
- `GET /ask`: Takes a query parameter `q` with the user's question and returns a JSON object containing the original query and the answer. Currently, it delegates to a placeholder `ask_agent` function defined in `agent_retriever_http_fix.py`. This function simply returns a default message because the agent implementation has not been completed.
- `GET /health`: Returns a simple JSON object to confirm the service is running.

There are also placeholders for ingesting data into the agent (`agent_ingest_index.py`) and retrieving answers via a more robust retriever. These modules are currently stubs and can be extended.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Stealth-mode-OFF/ai-book-agent.git
   cd ai-book-agent
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the FastAPI server using uvicorn:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 10000
   ```

## Environment Variables

The service is designed to work with external APIs and databases in the future. Environment variables are used to store sensitive configuration such as API keys and URLs. Do **not** hard code API keys in source files. Configure the following variables when deploying:

- `OPENAI_API_KEY`: Your OpenAI API key. This key should be provided in your deployment environment or secret manager. **Never commit your API keys to the repository.**
- `WEAVIATE_URL`: The URL of your Weaviate instance if you plan to use vector storage. In the provided `render.yaml` example, this is set to a placeholder value.

## Development Notes

- `agent_ingest_index.py`: Contains a TODO for implementing ingestion of documents or data into the agent's index.
- `agent_retriever_http_fix.py`: Provides the `ask_agent` function used by the `/ask` endpoint. Replace the default return value with your logic for querying your agent.
- `render.yaml`: Deployment configuration for Render.com. You can modify this file to match your hosting environment.

Feel free to extend the repository by implementing the ingestion and retrieval logic, integrating with vector databases like Weaviate, or connecting to language models such as OpenAI's GPT. Remember to keep secrets in environment variables or secret management systems.
