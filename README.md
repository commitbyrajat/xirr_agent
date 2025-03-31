# Using `langgraph-bigtool` to Calculate XIRR with an AI Agent

## Introduction

Calculating the Extended Internal Rate of Return (XIRR) is a key task in financial analysis, especially when dealing with irregular cash flows. Traditionally, developers would write custom numerical methods to compute XIRR. But with AI-powered tools, we can now offload this computation to intelligent agents, making the process more dynamic and scalable.

In this article, we’ll explore how to use the `langgraph-bigtool` library to build an AI agent that calculates XIRR efficiently, leveraging large language models (LLMs) and tool-based execution.

## Tech Stack

For this implementation, we’ll use:
- **Python 3.8+**
- **LangGraph** (for agent-based workflow execution)
- **LangGraph-BigTool** (for scalable tool execution)
- **LangChain** (for integrating LLMs)
- **OpenAI GPT-4o** or **Ollama Granite** (for AI-driven decision-making)
- **SciPy** (for numerical computation)
- **Dateutil** (for date parsing)

## Why Combine AI with Financial Tools?

LLMs are great at reasoning and understanding language, but they aren’t designed for precise numerical calculations. That’s where specialized tools like `scipy.optimize.newton` come in. By combining AI with tool-based execution, we can:
1. **Dynamically select the right tools** for the job.
2. **Perform mathematical operations with high accuracy.**
3. **Eliminate AI hallucinations** by enforcing tool-based computation.

## Understanding XIRR and Its Relationship with XNPV

XIRR (Extended Internal Rate of Return) helps determine the annualized return for cash flows occurring at different times. It’s widely used in investment analysis, especially when cash inflows and outflows aren’t evenly spaced.

XIRR relies on the XNPV (Extended Net Present Value) function. XNPV calculates the present value of cash flows discounted at a given rate, and XIRR is the discount rate at which XNPV equals zero.

### XNPV Formula:

$$XNPV = \sum \frac{C_i}{(1 + r)^{(t_i - t_0)/365}}$$

where:
- $C_i$ is the cash flow at time $t_i$
- $r$ is the discount rate
- $t_0$ is the date of the first cash flow

### XIRR Calculation:

To compute XIRR, we use numerical root-finding techniques like Newton’s method to find the discount rate \(r\) where XNPV equals zero.

## Implementing the XIRR Calculation Agent

### 1. Defining the Calculation Functions

We start by implementing functions to compute XNPV and XIRR using `SciPy`:

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

### 2. Integrating with `langgraph-bigtool`

Next, we register these functions as tools within `langgraph-bigtool`, enabling the AI agent to invoke them dynamically:

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

### 3. Running an XIRR Calculation Query

Now, we can use our AI agent to compute XIRR:

```python
query = """
I invested ₹10,000 on January 1, 2020. Over the next few years, I received:
- ₹2,000 on January 1, 2021  
- ₹4,000 on January 1, 2022  
- ₹6,000 on January 1, 2023  
- ₹8,000 on January 1, 2024  

Calculate my XIRR and return the results in JSON format.
"""
```

### 4. Agent Execution and Results

When executed, the AI agent selects the right tools and calculates:

```json
{
  "XNPV": 5092.02,
  "XIRR": "27.24%"
}
```

## Why `langgraph-bigtool`?
- **Scalability**: Handles multiple and large number of tools dynamically.
- **Modular Design**: Easily add new financial functions.
- **Accurate Execution**: Ensures precise calculations via specialized numerical methods.
- **Intelligent Decision-Making**: LLMs focus on reasoning, while tools handle computation.

## Conclusion
By leveraging `langgraph-bigtool`, we built a smart, scalable, and accurate XIRR calculator that dynamically selects and executes financial tools. This approach reduces manual effort, enhances reliability, and enables AI agents to perform real-world financial analysis with confidence.

