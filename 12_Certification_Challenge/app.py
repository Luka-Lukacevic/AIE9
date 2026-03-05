"""Chainlit UI for the Personal Finance Assistant."""

import chainlit as cl
from langchain_core.messages import HumanMessage

from agent import finance_agent


@cl.on_chat_start
async def start():
    cl.user_session.set("thread_id", cl.context.session.id)
    await cl.Message(
        content=(
            "Welcome to the **Personal Finance Assistant**!\n\n"
            "I can help you with:\n"
            "- Budgeting, saving, and debt management\n"
            "- Investing basics (stocks, bonds, mutual funds)\n"
            "- Currency conversion (live rates)\n"
            "- Financial goal-setting and scam awareness\n\n"
            "Ask me anything about personal finance!"
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}

    msg = cl.Message(content="")
    await msg.send()

    response = finance_agent.invoke(
        {"messages": [HumanMessage(content=message.content)]},
        config,
    )

    msg.content = response["messages"][-1].content
    await msg.update()
