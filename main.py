import os
from ollama import Client
from ollama import ChatResponse

# ANSI color codes
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


class ChatApplication:
    """Interactive chat application with Ollama."""

    def __init__(
        self, ollama_host: str = "http://127.0.0.1:11434", model: str = "llama3.2:3b"
    ):
        """Initialize the chat application with Ollama client and configuration."""
        self.client = Client(host=ollama_host)
        self.model = model
        self.ollama_host = ollama_host
        self.messages = []
        self.max_recent = 10
        self.threshold = 20

    def _get_system_prompt(self) -> dict:
        """Return the system prompt that guides the assistant's behavior."""
        return {
            "role": "system",
            "content": (
                "You are a helpful and friendly assistant with a great sense of humor. Your goal is to provide "
                "accurate, thoughtful responses while keeping things light and engaging. Always:\n"
                "- Be clear and concise\n"
                "- Don't be afraid to be a bit witty or use casual language when appropriate\n"
                "- Admit when you don't know something\n"
                "- Ask clarifying questions if needed\n"
                "- Maintain a respectful and approachable tone\n"
                "- Help people while making the conversation enjoyable"
            ),
        }

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
            messages=[{"role": "user", "content": summary_prompt}],
            stream=False,
        )

        return {
            "role": "system",
            "content": f"[Earlier conversation summary]: {response.message.content}",
        }

    def _manage_conversation_history(self) -> list:
        """Manage conversation history by summarizing old messages when threshold is reached."""
        system_prompt = self._get_system_prompt()

        if len(self.messages) <= self.threshold:
            return [system_prompt] + self.messages

        # Keep the last max_recent messages as-is
        recent_messages = self.messages[-self.max_recent :]
        old_messages = self.messages[: -self.max_recent]

        # Summarize old messages in batches to preserve mid-conversation context
        if old_messages:
            summary = self._summarize_messages(old_messages)
            if summary:
                return [system_prompt, summary] + recent_messages

        return [system_prompt] + recent_messages

    def _send_message(self, context_messages: list) -> str:
        """Send messages to Ollama and get a response."""
        response: ChatResponse = self.client.chat(
            model=self.model,
            messages=context_messages,
        )
        return response.message.content

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self.messages.append({"role": "assistant", "content": content})

    def chat(self, user_input: str) -> str:
        """Process user input and return assistant response."""
        self.add_user_message(user_input)
        context_messages = self._manage_conversation_history()
        assistant_message = self._send_message(context_messages)
        self.add_assistant_message(assistant_message)
        return assistant_message

    def run(self) -> None:
        """Run the interactive chat loop."""
        print(f"Chat Application (Connected to {self.ollama_host} using {self.model})")
        print("Type 'quit' or 'exit' to end the conversation.\n")

        while True:
            try:
                # Get user input
                user_input = input(f"{GREEN}❯{RESET} ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break

                # Get response from chat
                assistant_message = self.chat(user_input)
                print(f"\n{CYAN}{BOLD}Assistant:{RESET}\n{assistant_message}\n")

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


def main():
    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    app = ChatApplication(ollama_host=ollama_host, model=model)
    app.run()


if __name__ == "__main__":
    main()
