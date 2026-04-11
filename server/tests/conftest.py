import pytest_asyncio

from tests.helpers import close_registered_clients


@pytest_asyncio.fixture(autouse=True)
async def close_memory_clients():
    yield
    await close_registered_clients()
