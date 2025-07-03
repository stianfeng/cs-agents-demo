import httpx
from collections.abc import AsyncGenerator

from schema import UserInput, ChatMessage


class APIClient:
    """Client for interacting with the API service"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def astream(self, user_input: UserInput) -> AsyncGenerator[ChatMessage]:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", self.base_url + "/astream", json=user_input.model_dump()) as response:
                response.raise_for_status()
                async for chunk in response.aiter_text():
                    yield ChatMessage.model_validate_json(chunk.strip())
