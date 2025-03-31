import math
import types
import uuid

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import convert_positional_only_function_to_tool

llm = init_chat_model("openai:gpt-4o-mini")

all_tools = []
for function_name in dir(math):
    function = getattr(math, function_name)
    if not isinstance(function, types.BuiltinFunctionType):
        continue

    if tool := convert_positional_only_function_to_tool(function):
        all_tools.append(tool)

tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}

embeddings = init_embeddings("openai:text-embedding-3-small")

store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
        "fields": ["description"],
    }
)

for tool_id, tool in tool_registry.items():
    store.put(
        ("tools",),
        tool_id,
        {"description": f"{tool.name}: {tool.description}"},
    )

builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)


def invoke(query: str):
    for step in agent.stream(
        {"messages": query},
        stream_mode="updates",
    ):
        for _, update in step.items():
            for message in update.get("messages", []):
                message.pretty_print()
