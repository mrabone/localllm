# LocalLLM

Personal project where I experiment with LLMs locally.

## Features

- **Interactive CLI chat interface** - Have conversations with a local LLM in your terminal.
- **RAG Pipeline** - Populate a vector database with content from a list of websites.
- **Conversation history management** - Automatically summarizes old messages when conversation gets long to maintain context.
- **Customizable system prompt** - The assistant has a friendly, helpful personality.
- **No internet required** - Everything runs locally with Docker and Ollama.
- **Flexible model selection** - Easy to switch between different Ollama models.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- `uv` (recommended for dependency management)

### Running Locally with Docker

1.  **Configure Environment Variables:**
    
      Create a `.env` file from the example template:

      ```bash
      cp .env.example .env
      ```
      
      You can then modify your `.env` file to change the default settings where necessary.

2.  **Start Docker Services:**
    ```bash
    make setup
    ```
    This will start the Ollama and PostgreSQL containers.

3.  **Install Dependencies:**
    ```bash
    make sync-deps
    ```

4.  **Run the RAG Pipeline:**
         
      To populate the vector database with the content from the test data file (`rag/src/rag/data/reading_list_test_data.json`):

      ```bash
      make run-rag
      ```

5.  **Run the CLI Chat Application:**
    ```bash
    make run-cli
    ```
    The CLI will connect to the local Ollama instance and you can start chatting!

### Stopping the Services

```bash
docker-compose down
```

## Configuration

The application is configured via environment variables, which can be set in a `.env` file in the project root. An `.env.example` file is provided as a template.

### Common

-   `OLLAMA_BASE_URL`: The base URL for the Ollama API.

### RAG Pipeline

-   `PG_HOST`: The hostname of the PostgreSQL database.
-   `PG_PORT`: The port of the PostgreSQL database.
-   `PG_DATABASE`: The name of the database to use.
-   `PG_USER`: The username for the database.
-   `PG_PASSWORD`: The password for the database.
-   `PG_COLLECTION_NAME`: The name of the collection (table) to store the embeddings in.
-   `RAG_OLLAMA_MODEL`: The Ollama model to use for generating embeddings.
-   `CONCURRENT_REQUESTS`: The number of concurrent requests to make when scraping websites.
-   `REQUEST_DELAY`: The delay in seconds between requests.

### CLI Application

-   `CLI_OLLAMA_MODEL`: The Ollama model to use for the chat application.
-   `CLI_MAX_RECENT`: The number of recent messages to keep in the conversation history before summarizing.
-   `CLI_THRESHOLD`: The number of messages to keep in the conversation history before summarizing.

## Development

### Dependency Management

The project is organized as a workspace with multiple members (`cli`, `rag`). Dependencies are defined in the `pyproject.toml` file for each project.

To install these dependencies, run:
```bash
make sync-deps
```

### Changing the Model

1.  **Find available models** on [Ollama's model library](https://ollama.com/library).

2.  **Pull the model into your container:**
    ```bash
    docker compose exec ollama ollama pull <model-name>
    ```
    For example:
    ```bash
    docker compose exec ollama ollama pull llama2
    docker compose exec ollama ollama pull mistral
    ```

3.  **Update your `.env` file:**

      Change the `CLI_OLLAMA_MODEL` or `RAG_OLLAMA_MODEL` variables to the new model name in the `.env` file like so:

      ```
      CLI_OLLAMA_MODEL=<model-name>
      ```

4.  **Run the application:**
    ```bash
    make run-cli
    ```

