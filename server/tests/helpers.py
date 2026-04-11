from unittest.mock import MagicMock

import httpx
from mcp.types import TextContent, Tool

_open_clients: list[httpx.AsyncClient] = []


async def close_registered_clients() -> None:
    for client in _open_clients:
        await client.aclose()
    _open_clients.clear()


def _make_mcp_tool(
    name: str, description: str = "", params: dict | None = None
) -> Tool:
    """Build a minimal MCP Tool with the given name and input schema."""
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": params or {},
            "required": list((params or {}).keys()),
        },
    )


def _make_tool_result(text: str):
    """Return a mock MCP call_tool result with a real TextContent item."""
    content_item = TextContent(type="text", text=text)
    result = MagicMock()
    result.content = [content_item]
    return result


def _make_empty_tool_result():
    """Return a mock MCP call_tool result with no content."""
    result = MagicMock()
    result.content = []
    return result


def _make_stream_response(tokens: list[str]):
    """Return an async generator that yields mock stream chunks."""

    async def _gen():
        for token in tokens:
            chunk = MagicMock()
            chunk.message.content = token
            yield chunk

    return _gen()


def _make_memory_client(
    long_term_content: str = "",
    window_turns: list | None = None,
) -> httpx.AsyncClient:
    """Return an httpx.AsyncClient backed by a mock transport."""

    async def _transport(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/memory/long-term/" in path:
            return httpx.Response(200, json={"content": long_term_content})
        if "/memory/window/" in path:
            return httpx.Response(200, json=window_turns or [])
        if path == "/memory/messages":
            return httpx.Response(204)
        return httpx.Response(404)

    transport = httpx.MockTransport(_transport)
    client = httpx.AsyncClient(base_url="http://mcp-test", transport=transport)
    _open_clients.append(client)
    return client


def _make_client_with_transport(transport_fn) -> httpx.AsyncClient:
    """Return an httpx.AsyncClient with a custom transport, registered for cleanup."""
    client = httpx.AsyncClient(
        base_url="http://mcp-test",
        transport=httpx.MockTransport(transport_fn),
    )
    _open_clients.append(client)
    return client
