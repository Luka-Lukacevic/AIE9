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

# --- Persistent Vector Store Setup ---

QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_data")
os.makedirs(QDRANT_PATH, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_client = QdrantClient(path=QDRANT_PATH)

# Check if collection exists; create only if it doesn't
COLLECTION_NAME = "finance_knowledge"
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping ingestion.")
    collection_exists = True
except Exception:
    collection_exists = False

if not collection_exists:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks from {len(PDF_PATHS)} PDFs")
else:
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    print(f"Using existing collection with {len(chunks)} chunks")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

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
        response = httpx.get(url, timeout=10, verify=False)
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


@tool
def compound_interest_calculator(
    principal: float,
    annual_rate: float,
    years: float,
    monthly_contribution: float = 0,
) -> str:
    """Calculate compound interest with optional monthly contributions.

    Args:
        principal: Initial investment amount.
        annual_rate: Annual interest rate as a percentage (e.g., 7 for 7%).
        years: Number of years to invest.
        monthly_contribution: Optional monthly contribution amount (default 0).
    """
    try:
        r = annual_rate / 100  # Convert percentage to decimal
        n = 12  # Compounding monthly

        # Future value of principal with compound interest
        fv_principal = principal * (1 + r / n) ** (n * years)

        # Future value of monthly contributions (annuity)
        if monthly_contribution > 0:
            fv_contributions = monthly_contribution * (
                ((1 + r / n) ** (n * years) - 1) / (r / n)
            )
        else:
            fv_contributions = 0

        total_value = fv_principal + fv_contributions
        total_contributions = principal + (monthly_contribution * 12 * years)
        interest_earned = total_value - total_contributions

        return (
            f"Compound Interest Calculation:\n"
            f"- Initial Principal: ${principal:,.2f}\n"
            f"- Monthly Contribution: ${monthly_contribution:,.2f}\n"
            f"- Annual Rate: {annual_rate}%\n"
            f"- Time Period: {years} years\n\n"
            f"Results:\n"
            f"- Final Value: ${total_value:,.2f}\n"
            f"- Total Contributions: ${total_contributions:,.2f}\n"
            f"- Interest Earned: ${interest_earned:,.2f}\n"
            f"- Return on Investment: {(interest_earned / total_contributions * 100):.1f}%"
        )
    except Exception as e:
        return f"Error calculating compound interest: {e}"


@tool
def retirement_planner(
    current_age: int,
    retirement_age: int,
    current_savings: float,
    monthly_savings: float,
    expected_annual_return: float = 7.0,
    desired_monthly_income: float = 0,
) -> str:
    """Calculate retirement savings projections and provide recommendations.

    Args:
        current_age: Current age in years.
        retirement_age: Desired retirement age.
        current_savings: Current retirement savings amount.
        monthly_savings: Monthly contribution to retirement savings.
        expected_annual_return: Expected annual investment return percentage (default 7%).
        desired_monthly_income: Desired monthly income in retirement (optional).
    """
    try:
        years_to_retirement = retirement_age - current_age
        if years_to_retirement <= 0:
            return "Retirement age must be greater than current age."

        r = expected_annual_return / 100
        n = 12  # Monthly compounding

        # Future value of current savings
        fv_current = current_savings * (1 + r / n) ** (n * years_to_retirement)

        # Future value of monthly contributions
        fv_contributions = monthly_savings * (
            ((1 + r / n) ** (n * years_to_retirement) - 1) / (r / n)
        )

        total_at_retirement = fv_current + fv_contributions

        # Calculate sustainable monthly withdrawal (4% rule)
        annual_withdrawal = total_at_retirement * 0.04
        monthly_withdrawal = annual_withdrawal / 12

        result = (
            f"Retirement Plan Analysis:\n"
            f"- Current Age: {current_age}\n"
            f"- Retirement Age: {retirement_age}\n"
            f"- Years to Retirement: {years_to_retirement}\n"
            f"- Current Savings: ${current_savings:,.2f}\n"
            f"- Monthly Savings: ${monthly_savings:,.2f}\n"
            f"- Expected Return: {expected_annual_return}%\n\n"
            f"Projected Savings at Retirement: ${total_at_retirement:,.2f}\n\n"
            f"Estimated Sustainable Monthly Income (4% rule): ${monthly_withdrawal:,.2f}"
        )

        if desired_monthly_income > 0:
            gap = desired_monthly_income - monthly_withdrawal
            if gap > 0:
                # Calculate required monthly savings to meet goal
                required_savings = (
                    gap
                    * 12
                    / (((1 + r / n) ** (n * years_to_retirement) - 1) / (r / n))
                )
                result += (
                    f"\n\nGoal Gap Analysis:\n"
                    f"- Desired Monthly Income: ${desired_monthly_income:,.2f}\n"
                    f"- Gap: ${gap:,.2f}/month\n"
                    f"- To meet your goal, increase monthly savings by: ${required_savings:,.2f}"
                )
            else:
                result += (
                    f"\n\nGreat news! Your projected income exceeds your goal by ${abs(gap):,.2f}/month."
                )

        return result
    except Exception as e:
        return f"Error calculating retirement plan: {e}"


@tool
def debt_payoff_calculator(
    balance: float,
    apr: float,
    monthly_payment: float,
) -> str:
    """Calculate debt payoff timeline and total interest paid.

    Args:
        balance: Current outstanding balance.
        apr: Annual percentage rate (e.g., 18.99 for 18.99%).
        monthly_payment: Monthly payment amount.
    """
    try:
        if monthly_payment <= 0:
            return "Monthly payment must be greater than 0."

        monthly_rate = apr / 100 / 12

        # Check if payment covers interest
        if monthly_rate > 0 and monthly_payment <= balance * monthly_rate:
            return (
                f"Warning: Your monthly payment of ${monthly_payment:.2f} does not cover "
                f"the monthly interest of ${balance * monthly_rate:.2f}. "
                f"The debt will never be paid off. Increase your monthly payment."
            )

        months = 0
        current_balance = balance
        total_interest = 0

        while current_balance > 0 and months < 600:  # Cap at 50 years
            interest = current_balance * monthly_rate
            total_interest += interest
            principal = monthly_payment - interest
            current_balance -= principal
            months += 1

            if current_balance < 0:
                current_balance = 0

        years = months // 12
        remaining_months = months % 12

        total_paid = balance + total_interest

        return (
            f"Debt Payoff Analysis:\n"
            f"- Starting Balance: ${balance:,.2f}\n"
            f"- APR: {apr}%\n"
            f"- Monthly Payment: ${monthly_payment:,.2f}\n\n"
            f"Payoff Timeline: {years} years and {remaining_months} months\n"
            f"Total Interest Paid: ${total_interest:,.2f}\n"
            f"Total Amount Paid: ${total_paid:,.2f}\n"
            f"Interest as % of Principal: {(total_interest / balance * 100):.1f}%"
        )
    except Exception as e:
        return f"Error calculating debt payoff: {e}"


# --- User Profile Memory ---

import json
from pathlib import Path

PROFILE_FILE = Path("user_profiles.json")

def _load_profiles() -> dict:
    if PROFILE_FILE.exists():
        try:
            with open(PROFILE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def _save_profiles(profiles: dict):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profiles, f, indent=2)


@tool
def save_user_profile(
    user_id: str,
    age: int = 0,
    risk_tolerance: str = "",
    financial_goal: str = "",
    annual_income: float = 0,
    current_savings: float = 0,
) -> str:
    """Save user financial profile for personalized advice.

    Args:
        user_id: Unique identifier for the user.
        age: User's current age.
        risk_tolerance: Risk tolerance level (conservative/moderate/aggressive).
        financial_goal: Primary financial goal (e.g., "retirement", "buy a house", "emergency fund").
        annual_income: Annual income amount.
        current_savings: Current total savings amount.
    """
    try:
        profiles = _load_profiles()
        profile = {
            "age": age,
            "risk_tolerance": risk_tolerance.lower() if risk_tolerance else "",
            "financial_goal": financial_goal,
            "annual_income": annual_income,
            "current_savings": current_savings,
        }
        profiles[user_id] = profile
        _save_profiles(profiles)
        return f"Profile saved for user '{user_id}'. I can now provide personalized advice based on your financial situation."
    except Exception as e:
        return f"Error saving profile: {e}"


@tool
def get_user_profile(user_id: str) -> str:
    """Retrieve user financial profile for personalized advice.

    Args:
        user_id: Unique identifier for the user.
    """
    try:
        profiles = _load_profiles()
        profile = profiles.get(user_id)
        if not profile:
            return f"No profile found for user '{user_id}'. Please save your profile first."

        return (
            f"User Profile for '{user_id}':\n"
            f"- Age: {profile.get('age', 'Not set')}\n"
            f"- Risk Tolerance: {profile.get('risk_tolerance', 'Not set')}\n"
            f"- Financial Goal: {profile.get('financial_goal', 'Not set')}\n"
            f"- Annual Income: ${profile.get('annual_income', 0):,.2f}\n"
            f"- Current Savings: ${profile.get('current_savings', 0):,.2f}"
        )
    except Exception as e:
        return f"Error retrieving profile: {e}"


tools = [
    search_finance_knowledge,
    convert_currency,
    compound_interest_calculator,
    retirement_planner,
    debt_payoff_calculator,
    save_user_profile,
    get_user_profile,
]

# --- Agent Graph ---

FINANCE_SYSTEM_PROMPT = """You are a Personal Finance Education Assistant. You ONLY answer questions related to finance in general (educational), personal finance, budgeting, saving, investing, debt management, retirement planning, currency exchange, and financial calculations.

CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS:
For ANY question about financial concepts (stocks, bonds, investing, budgeting, emergency funds, etc.), you MUST use the search_finance_knowledge tool FIRST before answering. Do NOT answer from your general knowledge. The knowledge base contains authoritative information from CFPB and SEC documents that you must use.

Your role:
1. For questions about financial concepts (what is X, how does Y work, explain Z): ALWAYS call search_finance_knowledge tool FIRST, then answer based ONLY on what the tool returns.
2. For currency conversion requests, use the convert_currency tool.
3. For calculations (compound interest, retirement planning, debt payoff), use the appropriate calculator tools.
4. Keep explanations educational and beginner-friendly.
5. If the user has saved a profile, use get_user_profile to retrieve it and provide personalized advice.

Response format (ALWAYS follow this structure):
- **Explanation**: Clear, plain-language educational answer based on the search results.
- **Sources**: Cite the specific document name and page number from the search results (e.g., "sec-guide-to-savings-and-investing.pdf, p.12").
- **Disclaimer**: End every finance-related answer with:
  "Disclaimer: This information is for educational purposes only and does not constitute financial advice. Please consult a qualified financial advisor for personalized guidance."

If the search returns no results, say: "I couldn't find specific information about this in my knowledge base. Let me provide a general explanation, but please verify this with official sources."

GUARDRAILS:
- If the user asks about topics unrelated to finance, politely decline.
- Do not provide investment advice that suggests specific stocks, funds, or timing the market."""


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
