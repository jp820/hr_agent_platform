import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver 

from hr_agent_platform.scripts.nodes import (
    intent_router,
    policy_node,
    leave_node,
    travel_node,
    approval_node,
    route_intent,
    AgentState,
)

## BUILD GRAPH
builder = StateGraph(AgentState)

## Nodes addition in the graph
builder.add_node("intent_router", intent_router)
builder.add_node("policy_node", policy_node)
builder.add_node("leave_node", leave_node)
builder.add_node("travel_node", travel_node)
builder.add_node("approval_node", approval_node)

## Setting entry point for the graph
builder.set_entry_point("intent_router")

import logging
logger = logging.getLogger(__name__)

## Conditional edge for intent router in the graph
def route_intent(state: AgentState):
    intent = state.get("intent")
    logger.info(f"Routing to intent agent: {intent}")
    if intent == "leave":
        return "leave_node"
    elif intent == "travel":
        return "travel_node"
    elif intent == "other":
        return END
    else:
        return "policy_node"

builder.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "policy_node": "policy_node",
        "leave_node": "leave_node",
        "travel_node": "travel_node",
        END: END
    },
)

## Conditional edge for travel node in the graph
def route_travel(state: AgentState):
    if state.get("travel_request") and state["travel_request"].get("selected_flight"):
        # Check if it was already submitted in this turn's response
        if "submitted for approval" in (state.get("response") or ""):
            logger.info("Routing from Travel node: Selection already submitted, routing to END.")
            return END
        logger.info("Routing from Travel node: Selection made, routing to Approval node.")
        return "approval_node"
    logger.info("Routing from Travel node: Missing info or no selection, routing to END.")
    return END

builder.add_conditional_edges(
    "travel_node",
    route_travel,
    {
        "approval_node": "approval_node",
        END: END,
    },
)
builder.add_edge("approval_node", END)
builder.add_edge("policy_node", END)
builder.add_edge("leave_node", END)

# Add SQLite checkpointer for memory
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

graph = builder.compile(checkpointer=memory)

from langchain_core.messages import HumanMessage

# RUN EXAMPLES
import time

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test multi-turn interactive travel
    unique_id = str(int(time.time()))
    config = {"configurable": {"thread_id": f"test_{unique_id}"}}
    
    turns = [
        "I need to book travel to Kolkata",
        "From Mumbai on 2026-11-06",
        "It is a round trip. Return on 2026-11-10.",
        "Option 1 for outbound",
        "Option 2 for return",
        "Show my travel history"
    ]

    for turn in turns:
        print(f"\nUSER: {turn}")
        result = graph.invoke({
            "messages": [HumanMessage(content=turn)],
            "uid": "EMP001"
        }, config=config)
        print(f"AGENT: {result['response']}")


## Sample DB added
# Employee ID     Name             Annual Leave    Sick Leave
# 123             John Doe         5               10
# EMP001          Alice Smith      25              12
# EMP002          Bob Jones        15              8
# EMP003          Charlie Brown    20              10
# EMP004          David Wilson     10              5
# EMP005          Eve Davis        30              15
