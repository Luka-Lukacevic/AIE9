"""Shared agent logic for the Personal Finance Assistant.

This module is imported by both app.py (Chainlit) and api.py (FastAPI).
"""

import os
from typing import Annotated, Literal, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()

# --- Document Ingestion ---

PDF_PATHS = [
    "data/cfpb_your-money-your-goals_financial-empowerment_toolkit.pdf",
    "data/sec-guide-to-savings-and-investing.pdf",
]

all_docs = []
for path in PDF_PATHS:
    all_docs.extend(PyMuPDFLoader(path).load())

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(all_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(":memory:")
qdrant_client.create_collection(
    collection_name="finance_knowledge",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="finance_knowledge",
    embedding=embeddings,
)
vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

print(f"Ingested {len(chunks)} chunks from {len(PDF_PATHS)} PDFs")

# --- Tools ---


@tool
def search_finance_knowledge(query: str) -> str:
    """Search the personal finance knowledge base for information about budgeting,
    saving, investing, debt management, emergency funds, financial goals, and scams.

    Args:
        query: The search query to find relevant financial information.
    """
    results = retriever.invoke(query)
    if not results:
        return "No relevant information found in the finance knowledge base."
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Source {i}: {source}, p.{page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using live exchange rates.

    Args:
        amount: The amount of money to convert.
        from_currency: The source currency code (e.g. USD, EUR, GBP).
        to_currency: The target currency code (e.g. EUR, JPY, GBP).
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()
    try:
        url = f"https://api.frankfurter.dev/v1/latest?from={from_currency}&to={to_currency}"
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"][to_currency]
        converted = round(amount * rate, 2)
        return (
            f"{amount:,.2f} {from_currency} = {converted:,.2f} {to_currency}\n"
            f"Exchange rate: 1 {from_currency} = {rate} {to_currency}\n"
            f"Source: European Central Bank via Frankfurter API (date: {data.get('date', 'N/A')})"
        )
    except Exception as e:
        return f"Error fetching exchange rate: {e}"


tools = [search_finance_knowledge, convert_currency]

# --- Agent Graph ---

FINANCE_SYSTEM_PROMPT = """You are a Personal Finance Education Assistant.

Your role:
1. Answer personal finance questions using ONLY the knowledge base when available.
2. ALWAYS search the knowledge base for finance-related questions before answering.
3. For currency conversion requests, use the convert_currency tool.
4. Keep explanations educational and beginner-friendly.

Response format (ALWAYS follow this structure):
- **Explanation**: Clear, plain-language educational answer.
- **Sources**: Cite the document name and page number from retrieved context.
- **Disclaimer**: End every finance-related answer with:
  "Disclaimer: This information is for educational purposes only and does not constitute financial advice. Please consult a qualified financial advisor for personalized guidance."

If you cannot find relevant information, say so honestly."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState):
    messages = [SystemMessage(content=FINANCE_SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


tool_node = ToolNode(tools)
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

memory = MemorySaver()
finance_agent = workflow.compile(checkpointer=memory)
