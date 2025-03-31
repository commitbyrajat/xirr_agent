# Using `langgraph-bigtool` for Calculating XIRR with an AI Agent

## Introduction

Calculating the Extended Internal Rate of Return (XIRR) is a crucial operation in financial analysis. Traditional programming approaches require writing functions to compute XIRR using numerical methods. However, with the emergence of AI-driven agents, we can now utilize Large Language Models (LLMs) combined with tool-based execution to perform such computations dynamically.

In this article, we explore how to use the `langgraph-bigtool` library to create an agent that calculates XIRR efficiently.

## Tech Stack

Our implementation uses the following technologies:
- **Python 3.8+**
- **LangGraph** (for agent-based workflow execution)
- **LangGraph-BigTool** (for scalable tool execution)
- **LangChain** (for integrating LLMs)
- **OpenAI GPT-4o** or **Ollama Granite** (for AI-powered decision-making)
- **SciPy** (for numerical computation)
- **Dateutil** (for date parsing)

## Why Use Tools with LLMs?

While LLMs excel in reasoning and natural language processing, they lack precision in numerical computations. By integrating tools such as `scipy.optimize.newton` for root-finding, we enhance the AI’s capability to:
1. **Retrieve relevant tools dynamically** based on the problem context.
2. **Execute mathematical operations reliably** using predefined functions.
3. **Ensure correctness by enforcing tool-based computation**, avoiding AI hallucinations.

## How XIRR Works and the Role of XNPV

XIRR (Extended Internal Rate of Return) is used to calculate the annualized return for a set of cash flows occurring at irregular intervals. It is particularly useful for investments where cash inflows and outflows happen on different dates.

The calculation of XIRR depends on the XNPV (Extended Net Present Value) function. XNPV computes the present value of a series of cash flows discounted at a given rate. XIRR is determined by finding the discount rate at which XNPV equals zero.

### XNPV Formula:

$$XNPV = \sum \frac{C_i}{(1 + r)^{(t_i - t_0)/365}}$$
where:
- \( C$_i$ \) is the cash flow at time \( t$_i$ \)
- \( r \) is the discount rate
- \( t$_0$ \) is the date of the first cash flow

### XIRR Calculation:
XIRR is found using numerical methods such as Newton’s method to solve for \( r \) in the XNPV equation where \( XNPV = 0 \).

## Implementing the XIRR Calculation Agent

### 1. Define the XIRR Calculation Tools
We define functions for XNPV (Net Present Value) and XIRR calculations using `SciPy`:

```python
import json
from datetime import datetime
from dateutil import parser
from scipy.optimize import newton

def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")

def calculate_xnpv(discount_rate: float, cashflows_json: str) -> float:
    cashflows = json.loads(cashflows_json)
    cashflows = [(parse_date(entry["date"]), entry["amount"]) for entry in cashflows]
    initial_date = cashflows[0][0]
    return sum(
        amount / ((1 + discount_rate) ** ((date - initial_date).days / 365))
        for date, amount in cashflows
    )

def calculate_xirr(cashflows_json: str) -> float:
    cashflows = json.loads(cashflows_json)
    cashflows = [(parse_date(entry["date"]), entry["amount"]) for entry in cashflows]
    initial_date = cashflows[0][0]
    def xnpv_func(rate):
        return sum(
            amount / ((1 + rate) ** ((date - initial_date).days / 365))
            for date, amount in cashflows
        )
    return newton(xnpv_func, 0.1)  # Initial guess: 10%
```

### 2. Integrate the Tools with `langgraph-bigtool`
The `langgraph-bigtool` library allows our agent to dynamically search and invoke relevant tools.

```python
import types
import uuid
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore
from langgraph_bigtool import create_agent
from langgraph_bigtool.utils import convert_positional_only_function_to_tool
import tools.xirr.calculation

MODEL = "openai:gpt-4o-mini"
llm = init_chat_model(MODEL)
embeddings = init_embeddings("openai:text-embedding-3-small")

all_tools = []
for fxn_name in dir(tools.xirr.calculation):
    fxn = getattr(tools.xirr.calculation, fxn_name)
    if isinstance(fxn, types.FunctionType):
        if tool := convert_positional_only_function_to_tool(fxn):
            all_tools.append(tool)

tool_registry = {str(uuid.uuid4()): tool for tool in all_tools}
store = InMemoryStore(index={"embed": embeddings, "dims": 1536, "fields": ["description"]})
for tool_id, tool in tool_registry.items():
    store.put(("tools",), tool_id, {"description": f"{tool.name}: {tool.description}"})

builder = create_agent(llm, tool_registry)
agent = builder.compile(store=store)

def fxn_invoke(query: str):
    responses = []
    events = agent.stream({"messages": query}, stream_mode="values")
    for event in events:
        event["messages"][-1].pretty_print()
        responses.append(event["messages"][-1].content)
```

### 3. Execute an XIRR Calculation Query
We define an execution query to calculate XIRR dynamically using the agent:

```python
query = """
I have made the following investments and received returns on different dates:  
- Invested ₹10,000 on January 1, 2020  
- Received ₹2,000 on January 1, 2021  
- Received ₹4,000 on January 1, 2022  
- Received ₹6,000 on January 1, 2023  
- Received ₹8,000 on January 1, 2024  

### Instructions:
1. Analyze available tools before invoking them.
2. Calculate XNPV before calculating XIRR.
3. Use `parse_date` to convert dates.
4. Convert XIRR to percentage format.
5. Return output in JSON format:

'```json
{
  "XNPV": <calculated_xnpv_value>,
  "XIRR": "<calculated_xirr_value_in_percentage>%"
}
```'

"""
```

## Benefits of Using `langgraph-bigtool`
- **Scalability**: Supports hundreds of tools, making it highly adaptable.
- **Modular Architecture**: Allows easy addition of new tools without modifying existing logic.
- **Efficient Tool Retrieval**: Uses embeddings to fetch the best-suited tool dynamically.
- **Accurate Execution**: Ensures numerical accuracy by offloading computation to specialized tools.
- **Improved AI Reasoning**: LLMs focus on logical decision-making while tools handle computations.

## Conclusion
By leveraging `langgraph-bigtool`, we achieve:
- **Dynamic tool selection**: The agent identifies and retrieves relevant tools without hardcoding function calls.
- **Scalable execution**: We can add more financial computation tools without modifying the agent logic.
- **Accurate numerical computation**: Using specialized numerical methods ensures correctness over LLM-based estimations.

This approach is powerful for financial modeling, investment analysis, and automated portfolio optimization. 🚀
