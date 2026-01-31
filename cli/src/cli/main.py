from ollama import Client
from typing import Generator, Optional
from langchain_postgres import PGVector
from enum import Enum

from cli.config import settings
from cli.rag import get_rag_context, build_rag_prompt
from cli.services import ollama_client, pgvector_store


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ANSI color codes
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ChatApplication:
    """Interactive chat application with Ollama and optional RAG support."""

    def __init__(
        self,
        ollama_client: Client,
        pgvector_store: Optional[PGVector],
        model: str = settings.cli_ollama_model,
        max_recent: int = settings.cli_max_recent,
        threshold: int = settings.cli_threshold,
    ):
        """Initialize the chat application with required services and configuration."""
        self.client = ollama_client
        self.pgvector_store = pgvector_store
        self.model = model
        self.messages = []
        self.max_recent = max_recent
        self.threshold = threshold
        self.rag_enabled = pgvector_store is not None

    def _summarize_messages(self, messages_to_summarize: list) -> dict | None:
        """Summarize a list of messages into a single message."""
        if not messages_to_summarize:
            return None

        # Create a summary prompt
        conversation_text = "\n".join(
            [
                f"{msg['role'].title()}: {msg['content']}"
                for msg in messages_to_summarize
            ]
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

        return {
            "role": Role.SYSTEM.value,
            "content": f"[Earlier conversation summary]: {response['message']['content']}",
        }

    def _manage_conversation_history(self) -> list:
        """Manage conversation history by summarizing old messages when threshold is reached."""
        if len(self.messages) <= self.threshold:
            return self.messages

        # Keep the last max_recent messages as-is
        recent_messages = self.messages[-self.max_recent :]
        old_messages = self.messages[: -self.max_recent]

        # Summarize old messages in batches to preserve mid-conversation context
        if old_messages:
            summary = self._summarize_messages(old_messages)
            if summary:
                return [summary] + recent_messages

        return recent_messages

    def _send_message(self, context_messages: list) -> Generator[str, None, None]:
        """Send messages to Ollama and stream the response."""
        stream = self.client.chat(
            model=self.model, messages=context_messages, stream=True
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    def add_message(self, role: Role, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role.value, "content": content})

    def _prepare_and_send(self, message_content: str) -> Generator[str, None, None]:
        """Add a message to history, manage context, and send to the model."""
        self.add_message(Role.USER, message_content)
        context_messages = self._manage_conversation_history()
        return self._send_message(context_messages)

    def chat(self, user_input: str) -> Generator[str, None, None]:
        """Process user input and stream the assistant's response."""
        context = get_rag_context(user_input, self.pgvector_store)
        if not context:
            return self._prepare_and_send(user_input)

        enriched_input = build_rag_prompt(user_input, context)
        print(
            f"{CYAN}(RAG: Retrieved context from {context.count('[Document')} documents){RESET}"
        )
        return self._prepare_and_send(enriched_input)

    def run(self) -> None:
        """Run the interactive chat loop."""
        print(f"Chat Application (Model: {self.model})")
        if self.rag_enabled:
            print(f"RAG enabled (Collection: '{settings.pg_collection_name}')")
        print("Type 'quit' or 'exit' to end the conversation.\n")

        while True:
            try:
                user_input = input(f"{GREEN}❯{RESET} ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                print(f"\n{CYAN}{BOLD}Assistant:{RESET}")
                assistant_message = ""
                # Stream the response from the chat method
                stream = self.chat(user_input)
                for chunk in stream:
                    assistant_message += chunk
                    print(chunk, end="", flush=True)

                # Add the complete assistant message to the history
                self.add_message(Role.ASSISTANT, assistant_message)
                print("\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


def main():
    """Main entry point for the application."""
    try:
        app = ChatApplication(
            ollama_client=ollama_client,
            pgvector_store=pgvector_store,
        )
        app.run()
    except Exception as e:
        print(f"Error: Failed to start the application. Details: {e}")
        exit(1)


if __name__ == "__main__":
    main()
