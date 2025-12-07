# LocalLLM

Personal project where I experiment with LLMs locally.

## Features

- **Interactive CLI chat interface** - Have conversations with a local LLM in your terminal
- **Conversation history management** - Automatically summarizes old messages when conversation gets long to maintain context
- **Customizable system prompt** - The assistant has a friendly, helpful personality
- **No internet required** - Everything runs locally with Docker and Ollama
- **Flexible model selection** - Easy to switch between different Ollama models

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- `uv` (optional, but recommended for dependency management)

### Running Locally with Docker

1. **Start the Ollama service:**
   ```bash
   make setup
   ```
   Or alternatively with docker-compose:
   ```bash
   docker-compose up -d
   ```
   This will:
   - Start the Ollama container on `http://127.0.0.1:11434`
   - Automatically download and load the `llama3.2:3b` model
   - Display progress in the logs

   Monitor the download progress:
   ```bash
   docker logs localllm-ollama
   ```

   Wait for the message `✓ Model ready! You can now run your application` before going to the next step.

2. **Install Python dependencies:**
   ```bash
   uv sync
   ```
   Or alternatively with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the chat application:**
   ```bash
   uv run main.py
   ```
   Or alternatively with Python:
   ```bash
   python main.py
   ```

   The CLI will connect to the local Ollama instance and you can start chatting!

### Stopping the Container/LLM

```bash
docker-compose down
```

## Configuration

The application uses these environment variables (with defaults):

- `OLLAMA_HOST`: Ollama API endpoint (default: `http://127.0.0.1:11434`)
- `OLLAMA_MODEL`: Model to use (default: `llama3.2:3b`)
- `OLLAMA_CONTEXT_LENGTH`: Context window size for Ollama models (default: `4096`)

You can override these values by running:
```bash
OLLAMA_HOST=http://custom-host:11434 OLLAMA_MODEL=llama2 python main.py
```

### Changing the Model

1. **Find available models** on [Ollama's model library](https://ollama.com/library)

2. **Pull the model into your container:**
   ```bash
   docker exec localllm-ollama ollama pull <model-name>
   ```
   For example:
   ```bash
   docker exec localllm-ollama ollama pull llama2
   docker exec localllm-ollama ollama pull mistral
   ```

3. **Run the application with the new model:**
   ```bash
   OLLAMA_MODEL=mistral uv run main.py
   ```

Note: Larger models may require more resources. Check the model's requirements and adjust `docker-compose.yml` resource limits if needed.

## Docker Resource Limits

The `docker-compose.yml` includes resource constraints:
- **CPU**: 4 cores max, 2 reserved
- **Memory**: 8GB max, 4GB reserved

Adjust these in the `docker-compose.yml` file under the `deploy.resources` section if needed.
