import types
import uuid

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import convert_positional_only_function_to_tool

import tools.xirr.calculation

disable_openai = False
MODEL = "ollama:granite3.2:8b" if disable_openai else "openai:gpt-4o-mini"
EMBEDDER = (
    "ollama:nomic-embed-text:latest"
    if disable_openai
    else "openai:text-embedding-3-small"
)

print("MODEL: ", MODEL)
print("\nEMBEDDER: ", EMBEDDER)

llm = init_chat_model(MODEL)
embeddings = init_embeddings(EMBEDDER)


all_tools = []
for fxn_name in dir(tools.xirr.calculation):
    fxn = getattr(tools.xirr.calculation, fxn_name)
    if not isinstance(fxn, types.FunctionType):
        continue

    if tool := convert_positional_only_function_to_tool(fxn):
        all_tools.append(tool)

tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)

for tool_id, tool in tool_registry.items():
    store.put(("tools",), tool_id, {"description": f"{tool.name}: {tool.description}"})

builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)


def fxn_invoke(query: str):
    responses = []
    events = agent.stream({"messages": query}, stream_mode="values")

    for event in events:
        event["messages"][-1].pretty_print()
        response = event["messages"][-1].content
        responses.append(response)
