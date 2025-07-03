import logging
import json
from collections.abc import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from schema import UserInput
from agent import agent
from settings import settings


# Set up logging
logging.basicConfig(level=logging.INFO)

# Message generator for streaming
async def message_generator(user_input: UserInput) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": user_input.thread_id}}
    try:
        async for msg, metadata in agent.astream(
            {"messages": HumanMessage(content=user_input.message)},
            config,
            stream_mode="messages",
        ):
            if isinstance(msg, AIMessage) and metadata['langgraph_node'] != "tools":
                response = {
                    "type": "ai",
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "response_metadata": msg.response_metadata,
                }
                yield (json.dumps(response))
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                response = {
                    "type": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                    "response_metadata": msg.response_metadata,
                }
                yield (json.dumps(response))
    except Exception as e:
        response = {
            "type": "ai",
            "content": str(e),
        }
        logging.exception("Error during message generation")
        yield (json.dumps(response))


app = FastAPI()

@app.post("/astream", response_class=StreamingResponse)
async def astream(user_input: UserInput) -> StreamingResponse:
    return StreamingResponse(
        message_generator(user_input), 
        media_type="application/x-ndjson"
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT, log_level="debug")