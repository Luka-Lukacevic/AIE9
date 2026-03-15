"""An agent graph with a clarity and conciseness evaluation node.

After the agent responds, a secondary node evaluates if the response is clear and concise.
If it meets the criteria, end; otherwise, ask the agent to improve it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

from app.state import MessagesState
from app.models import get_chat_model
from app.tools import get_tool_belt


class ClarityResult(BaseModel):
    is_clear: bool = Field(description="Whether the response is clear, concise, and well-structured")
    feedback: str = Field(description="Specific feedback on what could be improved if not clear")


def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: MessagesState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


def route_to_action_or_clarity(state: MessagesState):
    """Decide whether to execute tools or run the clarity evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "clarity"


_clarity_prompt = ChatPromptTemplate.from_template(
    "Evaluate the following response for clarity and conciseness. "
    "A good response should be:\n"
    "- Clear and easy to understand\n"
    "- Concise without unnecessary words\n"
    "- Well-structured with logical flow\n"
    "- Directly addresses the question\n\n"
    "User Question:\n{user_question}\n\n"
    "Agent Response:\n{agent_response}"
)


def clarity_node(state: MessagesState) -> dict:
    """Evaluate clarity and conciseness of the latest response."""
    # Safety limit: prevent infinite loops
    if len(state["messages"]) > 12:
        return {"messages": [AIMessage(content="CLARITY:END")]}

    # Find the original user question (first human message)
    user_question = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and msg.role == "user":
            user_question = msg.content
            break
    
    if not user_question:
        user_question = "Unknown question"

    # Get the latest agent response
    agent_response = state["messages"][-1]
    response_content = getattr(agent_response, "content", str(agent_response))

    # Evaluate clarity using structured output
    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(ClarityResult)
    result = (_clarity_prompt | structured_model).invoke(
        {
            "user_question": user_question,
            "agent_response": response_content,
        }
    )

    messages_to_add = []
    
    # Add evaluation result marker
    decision = "Y" if result.is_clear else "N"
    messages_to_add.append(AIMessage(content=f"CLARITY:{decision}"))
    
    # If not clear, add feedback message for the agent to improve
    if not result.is_clear:
        improvement_request = HumanMessage(
            content=f"Please improve your previous response to be clearer and more concise. Specific feedback: {result.feedback}"
        )
        messages_to_add.append(improvement_request)
    
    return {"messages": messages_to_add}


def clarity_decision(state: MessagesState):
    """Terminate on 'CLARITY:Y' or loop back to improve; guard against infinite loops."""
    if any(getattr(m, "content", "") == "CLARITY:END" for m in state["messages"][-1:]):
        return END

    last = state["messages"][-1]
    text = getattr(last, "content", "")
    
    if "CLARITY:Y" in text:
        return "end"
    
    # If not clear, loop back to agent
    return "continue"


def build_graph():
    """Build an agent graph with a clarity and conciseness evaluation node."""
    graph = StateGraph(MessagesState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("clarity", clarity_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_clarity,
        {"action": "action", "clarity": "clarity"},
    )
    graph.add_conditional_edges(
        "clarity",
        clarity_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
