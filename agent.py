from typing import Annotated
from typing_extensions import TypedDict
import logging

from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from settings import settings
from tools import lookup_tncs, get_recommendation


# State of the graph
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# LLM of the assistant
if settings.OPENAI_API_KEY:
    llm = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.OPENAI_MODEL,
        temperature=0,
    )
    logging.info(f"Using OpenAI model: {settings.OPENAI_MODEL}")
else:
    llm = ChatOllama(
        model="cogito",
        temperature=0,
    )
    logging.info(f"Using Ollama model: {llm.model}")

prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful customer service assistant with two tools:\n"
            "- lookup_tncs: Use this tool to lookup terms and conditions related to either mobile or broadband.\n"
            "- get_recommendation: Use this tool to get a mobile plan recommendation based on the user's query.\n"
            "You can only use one tool at a time. Only use information from the tools to answer the user's question.\n",
        ),
        MessagesPlaceholder('messages'),
    ]
)

tools = [lookup_tncs, get_recommendation]
assistant = prompt | llm.bind_tools(tools)

def cs_agent(state: State):
    return {"messages": [assistant.invoke(state["messages"])]}

# Define workflow
workflow = StateGraph(State)
workflow.add_node('cs_agent', cs_agent)
workflow.add_node('tools', ToolNode(tools))
workflow.add_edge(START, 'cs_agent')
workflow.add_conditional_edges(
    'cs_agent',
    tools_condition,
)
workflow.add_edge('tools', 'cs_agent')

checkpointer = InMemorySaver()
agent = workflow.compile(checkpointer=checkpointer, name='cs_agent')