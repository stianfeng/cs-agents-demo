import streamlit as st
import uuid
from typing import List
import asyncio
from collections.abc import AsyncGenerator

from schema import UserInput, ChatMessage
from settings import settings
from api_client import APIClient


# Set API client
st.session_state.api_client = APIClient(f"http://{settings.API_HOST}:{settings.API_PORT}")

# Set page config
st.set_page_config(
    page_title="Customer Service Agent",
    layout="wide",
    menu_items={},
    initial_sidebar_state="expanded",
)

# Generate thread ID
def get_or_create_thread_id() -> str:
    """Get the order ID from session state or URL parameters, or create new one if it doesn't exist."""
    # Check if thread_id is in session state
    if "thread_id" in st.session_state:
        return st.session_state.thread_id

    # Check URL parameters
    if "thread_id" in st.query_params:
        thread_id = st.query_params["thread_id"]
        st.session_state.thread_id = thread_id
        return thread_id
    
    # Generate new thread_id
    thread_id = str(uuid.uuid4())
    # Store in session state and URL parameters
    st.session_state.thread_id = thread_id
    st.query_params.thread_id = thread_id
    return thread_id

# Set session state
thread_id = get_or_create_thread_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar display
with st.sidebar:
    if st.button(":material/chat: New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Draw messages
async def draw_messages(messages_agen: AsyncGenerator[ChatMessage], is_new: bool=False):
    # Keep track of last message drawn
    last_message_type = None
    st.session_state.last_message = None

    while msg := await anext(messages_agen, None):
        match msg.type:
            # Display human message
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # Display AI message
            case "ai":
                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                    streaming_placeholder = st.session_state.last_message.empty()
                    streaming_content = ""

                with st.session_state.last_message:
                    # Write content if not empty
                    if msg.content:
                        streaming_content += msg.content
                        streaming_placeholder.write(streaming_content)

                    # Status container to show tool calls
                    if msg.tool_calls:
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            status = st.status(
                                f"""Tool Call: {tool_call["name"]}""",
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status
                            status.write("Input:")
                            status.write(tool_call["args"])

            # Update tool call status
            case "tool":
                if is_new:
                    st.session_state.messages.append(msg)

                try:
                    status = call_results[msg.tool_call_id]
                    status.write("Output:")
                    status.write(msg.content)
                    status.update(state="complete")
                except Exception as e:
                    print(e)
                    st.error(f"Unexpected error: {e}")
                    return

async def main():
    # Draw existing messages
    messages: List[ChatMessage] = st.session_state.messages
    async def amessage_iter():
        for msg in messages:
            yield msg
    await draw_messages(amessage_iter())

    # Generate new messages after a user input
    if user_input := st.chat_input():
        st.session_state.messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)

        # Call APIClient and draw messages
        try:
            stream = st.session_state.api_client.astream(
                UserInput(
                    thread_id=st.session_state.thread_id,
                    message=user_input,
                )
            )
            await draw_messages(stream, is_new=True)
        except Exception as e:
            print(e)
            st.error(f"Unexpected error: {e}")
            return

asyncio.run(main())