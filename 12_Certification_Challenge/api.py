"""FastAPI backend for the Personal Finance Assistant."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import finance_agent

app = FastAPI(title="Personal Finance Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        config = {"configurable": {"thread_id": request.thread_id}}
        response = finance_agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config,
        )
        return {"reply": response["messages"][-1].content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
