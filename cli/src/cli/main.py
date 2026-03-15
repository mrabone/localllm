import argparse
import sys

import httpx
from httpx_sse import connect_sse

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

        print(f"\n{Colors.CYAN}{Colors.BOLD}Assistant:{Colors.RESET}")

        with connect_sse(
            self.client,
            "POST",
            url,
            json={"message": user_input},
            timeout=httpx.Timeout(connect=10.0, read=300.0),
        ) as event_source:
            if event_source.response.status_code != 200:
                # Truncate the body to avoid leaking server-side stack traces
                # or configuration details to the terminal.
                # .read() must be called explicitly because the response is
                # opened as a stream and .text is not available until buffered.
                event_source.response.read()
                raw = event_source.response.text[:200]
                print(f"Server error {event_source.response.status_code}: {raw}")
                return

            for sse in event_source.iter_sse():
                if sse.event == "token":
                    print(sse.data, end="", flush=True)
                elif sse.event == "error":
                    print(f"\nServer error: {sse.data}")
                    break
                elif sse.event == "done":
                    break

        print("\n")

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
