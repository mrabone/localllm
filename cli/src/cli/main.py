import argparse
import sys

import httpx

from cli.config import settings
from cli.services import get_or_create_session


class Colors:
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


class ChatApplication:
    """Thin CLI presentation layer.

    All chat logic (history management, RAG, summarisation) lives in the
    server.  This class is responsible only for reading user input, sending it
    to the server, and rendering the streaming response.
    """

    def __init__(self, client: httpx.Client, session_name: str | None = None) -> None:
        self.client = client
        self.session_id, self.session_name = get_or_create_session(
            client, name=session_name
        )

    def chat(self, user_input: str) -> None:
        """Send user_input to the server and stream the response to stdout."""
        url = f"{settings.server_url}/sessions/{self.session_id}/chat"

        with self.client.stream(
            "POST",
            url,
            json={"message": user_input},
            timeout=None,
        ) as response:
            if response.status_code != 200:
                print(f"Server error {response.status_code}: {response.text}")
                return

            rag_doc_count: int | None = None
            print(f"\n{Colors.CYAN}{Colors.BOLD}Assistant:{Colors.RESET}")

            # SSE frames arrive as pairs of "event: <name>" / "data: <value>"
            # lines separated by blank lines.  Track current event name so
            # that the data handler always has a valid event to act on.
            current_event: str = "message"
            for line in response.iter_lines():
                if not line:
                    current_event = "message"
                    continue

                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    data = line[len("data: ") :]

                    if current_event == "token":
                        print(data, end="", flush=True)
                    elif current_event == "rag":
                        rag_doc_count = int(data)
                    elif current_event == "error":
                        print(f"\nServer error: {data}")
                        break
                    elif current_event == "done":
                        break

            print("\n")

            if rag_doc_count is not None:
                print(
                    f"{Colors.CYAN}(RAG: retrieved context from "
                    f"{rag_doc_count} document(s)){Colors.RESET}\n"
                )

    def run(self) -> None:
        """Run the interactive chat REPL until the user exits."""
        label = f"'{self.session_name}'" if self.session_name else str(self.session_id)
        print(f"Chat Application (session: {label})")
        print("Type 'quit' or 'exit' to end the conversation.\n")

        while True:
            try:
                user_input = input(f"{Colors.GREEN}>{Colors.RESET} ").strip()
                if not user_input:
                    continue
                if user_input.lower() in {"quit", "exit"}:
                    print("Goodbye!")
                    break

                self.chat(user_input)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="localllm",
        description="Chat with a local LLM via the LocalLLM server.",
    )
    parser.add_argument(
        "--session",
        metavar="NAME",
        default=None,
        help=(
            "Name of the session to load or create. "
            "If omitted, a new anonymous session is started."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Initialise the HTTP client and start the interactive chat application."""
    args = _parse_args()
    try:
        with httpx.Client() as client:
            app = ChatApplication(client=client, session_name=args.session)
            app.run()
    except Exception as e:
        print(f"Failed to start the application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
