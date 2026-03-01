import sys
from enum import Enum
from typing import Generator

from langchain_postgres import PGVector
from ollama import Client

from cli.config import settings
from cli.rag import RagResult, build_rag_prompt, get_rag_context
from cli.services import get_ollama_client, get_rag_store


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Colors:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ChatApplication:
    """Interactive chat application with Ollama and optional RAG support."""

    def __init__(
        self,
        ollama_client: Client,
        pgvector_store: PGVector | None,
        model: str | None = None,
        max_recent: int | None = None,
        threshold: int | None = None,
    ) -> None:
        """Initialise the chat application with required services and configuration.

        Args:
            ollama_client: An initialised Ollama client.
            pgvector_store: An initialised PGVector store, or None to run without RAG.
            model: Ollama model name. Defaults to settings.cli_ollama_model.
            max_recent: Number of recent messages to keep verbatim before summarisation.
            threshold: Total message count that triggers history summarisation.
        """
        self.client = ollama_client
        self.pgvector_store = pgvector_store
        self.model = model if model is not None else settings.cli_ollama_model
        self.max_recent = (
            max_recent if max_recent is not None else settings.cli_max_recent
        )
        self.threshold = threshold if threshold is not None else settings.cli_threshold
        self.rag_enabled = pgvector_store is not None
        self.messages: list[dict[str, str]] = []

    def _summarize_messages(
        self, messages_to_summarize: list[dict[str, str]]
    ) -> dict[str, str] | None:
        """Summarize a list of messages into a single system message.

        Returns None if the input is empty or the model returns no content.
        """
        if not messages_to_summarize:
            return None

        conversation_text = "\n".join(
            f"{msg['role'].title()}: {msg['content']}" for msg in messages_to_summarize
        )
        summary_prompt = (
            "Summarize the following conversation concisely in 2-3 sentences, "
            "preserving key points and context:\n\n"
            f"{conversation_text}\n\n"
            "Summary:"
        )

        response = self.client.chat(
            model=self.model,
            messages=[{"role": Role.USER.value, "content": summary_prompt}],
            stream=False,
        )

        content = response.get("message", {}).get("content", "").strip()
        if not content:
            return None

        return {
            "role": Role.SYSTEM.value,
            "content": f"[Earlier conversation summary]: {content}",
        }

    def _manage_conversation_history(self) -> list[dict[str, str]]:
        """Return a context window, summarising old messages when the threshold is reached."""
        if len(self.messages) <= self.threshold:
            return self.messages

        recent_messages = self.messages[-self.max_recent :]
        old_messages = self.messages[: -self.max_recent]

        if old_messages:
            summary = self._summarize_messages(old_messages)
            if summary:
                return [summary] + recent_messages

        return recent_messages

    def add_message(self, role: Role, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({"role": role.value, "content": content})

    def _send_message(
        self, context_messages: list[dict[str, str]]
    ) -> Generator[str, None, None]:
        """Stream token chunks from Ollama for the given message list."""
        stream = self.client.chat(
            model=self.model, messages=context_messages, stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    def _prepare_and_send(self, message_content: str) -> Generator[str, None, None]:
        """Add a user message to history, trim context, and stream the response."""
        self.add_message(Role.USER, message_content)
        context_messages = self._manage_conversation_history()
        return self._send_message(context_messages)

    def chat(
        self, user_input: str
    ) -> tuple[Generator[str, None, None], RagResult | None]:
        """Process user input and return a (response_stream, rag_result) tuple.

        The rag_result is None when RAG is disabled or returned no useful context.
        The caller is responsible for rendering any RAG status to the user.
        """
        rag_result = get_rag_context(user_input, self.pgvector_store)
        if rag_result is None:
            return self._prepare_and_send(user_input), None

        enriched_input = build_rag_prompt(user_input, rag_result.context)
        return self._prepare_and_send(enriched_input), rag_result

    def run(self) -> None:
        """Run the interactive chat loop until the user exits."""
        print(f"Chat Application (Model: {self.model})")
        if self.rag_enabled:
            print(f"RAG enabled (Collection: '{settings.pg_collection_name}')")
        print("Type 'quit' or 'exit' to end the conversation.\n")

        while True:
            try:
                user_input = input(f"{Colors.GREEN}>{Colors.RESET} ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit"}:
                    print("Goodbye!")
                    break

                response_stream, rag_result = self.chat(user_input)

                if rag_result is not None:
                    print(
                        f"{Colors.CYAN}(RAG: retrieved context from "
                        f"{rag_result.document_count} document(s)){Colors.RESET}"
                    )

                print(f"\n{Colors.CYAN}{Colors.BOLD}Assistant:{Colors.RESET}")
                assistant_message = ""
                for chunk in response_stream:
                    assistant_message += chunk
                    print(chunk, end="", flush=True)

                self.add_message(Role.ASSISTANT, assistant_message)
                print("\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


def main() -> None:
    """Initialise services and start the interactive chat application."""
    try:
        app = ChatApplication(
            ollama_client=get_ollama_client(),
            pgvector_store=get_rag_store(),
        )
        app.run()
    except Exception as e:
        print(f"Failed to start the application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
