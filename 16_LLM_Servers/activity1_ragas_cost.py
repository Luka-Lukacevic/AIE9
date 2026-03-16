"""Activity 1 runner: Fireworks vs OpenAI RAGAS + LangSmith cost comparison.

Usage:
  uv run python activity1_ragas_cost.py

Required env vars:
  - FIREWORKS_API_KEY
  - OPENAI_API_KEY
  - LANGSMITH_API_KEY (for tracing/cost dashboard)

Optional env vars:
  - FIREWORKS_CHAT_MODEL (default: accounts/fireworks/models/gpt-oss-20b)
  - FIREWORKS_EMBEDDING_MODEL (default: accounts/fireworks/models/qwen3-embedding-4b)
  - OPENAI_CHAT_MODEL (default: gpt-4.1-mini)
  - OPENAI_EMBEDDING_MODEL (default: text-embedding-3-large)
  - RAG_DATA_DIR (default: data)
  - TOP_K (default: 4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import tiktoken
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _token_len(text: str) -> int:
    return len(tiktoken.encoding_for_model("gpt-4o").encode(text))


def _load_docs(data_dir: str) -> list[Document]:
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"No PDF documents found in {data_dir!r}")
    return docs


def _split_docs(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=50,
        length_function=_token_len,
    )
    return splitter.split_documents(docs)


def _fw_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.environ.get(
            "FIREWORKS_EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-4b"
        ),
        openai_api_key=_require_env("FIREWORKS_API_KEY"),
        openai_api_base="https://api.fireworks.ai/inference/v1",
        check_embedding_ctx_length=False,
    )


def _oa_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
    )


def _fw_chat() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"),
        openai_api_key=_require_env("FIREWORKS_API_KEY"),
        openai_api_base="https://api.fireworks.ai/inference/v1",
        temperature=0,
    )


def _oa_chat() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
    )


def _build_retriever(chunks: list[Document], embeddings: OpenAIEmbeddings):
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="activity1",
    )
    return vectorstore.as_retriever(search_kwargs={"k": int(os.environ.get("TOP_K", "4"))})


def _format_context(docs: Iterable[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _answer_question(llm: ChatOpenAI, question: str, contexts: list[str]) -> str:
    prompt = (
        "Use only the provided context to answer.\n"
        "If the answer is not in context, say exactly: I don't know.\n\n"
        f"Question:\n{question}\n\n"
        "Context:\n"
        + "\n\n".join(contexts)
    )
    return llm.invoke(prompt).content


def _build_reference_answers(questions: list[str], contexts: list[list[str]]) -> list[str]:
    # Use a strong model once to generate consistent references from retrieved context.
    judge = _oa_chat()
    references: list[str] = []
    for q, ctx in zip(questions, contexts):
        prompt = (
            "You are creating reference answers for RAG evaluation.\n"
            "Use only the supplied context. Be concise and factual.\n"
            'If the context is insufficient, answer exactly: I don\'t know.\n\n'
            f"Question:\n{q}\n\n"
            "Context:\n"
            + "\n\n".join(ctx)
        )
        references.append(judge.invoke(prompt).content)
    return references


@dataclass
class ProviderRun:
    name: str
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    references: list[str]


def _run_provider(
    *,
    name: str,
    questions: list[str],
    retriever,
    llm: ChatOpenAI,
    references: list[str],
) -> ProviderRun:
    contexts: list[list[str]] = []
    answers: list[str] = []

    for q in questions:
        docs = retriever.invoke(q)
        ctx = [d.page_content for d in docs]
        contexts.append(ctx)
        answers.append(_answer_question(llm, q, ctx))

    return ProviderRun(
        name=name,
        questions=questions,
        answers=answers,
        contexts=contexts,
        references=references,
    )


def _evaluate_with_ragas(run: ProviderRun) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_correctness, context_precision, faithfulness
    except Exception as exc:
        raise RuntimeError(
            "Missing RAGAS dependencies. Install with:\n"
            "  uv add ragas datasets\n"
        ) from exc

    ds = Dataset.from_dict(
        {
            "question": run.questions,
            "answer": run.answers,
            "contexts": run.contexts,
            "reference": run.references,
        }
    )

    result = evaluate(ds, metrics=[context_precision, faithfulness, answer_correctness])
    return result


def main() -> None:
    load_dotenv()

    # LangSmith tracing for token/cost dashboards.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    _require_env("LANGSMITH_API_KEY")
    _require_env("FIREWORKS_API_KEY")
    _require_env("OPENAI_API_KEY")

    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    docs = _load_docs(data_dir)
    chunks = _split_docs(docs)

    questions = [
        "Which core vaccines are recommended for kittens in the first months of life?",
        "At what age should rabies vaccination usually be given to kittens?",
        "How often should adult indoor cats receive wellness checkups?",
        "What are common warning signs that require urgent veterinary care in cats?",
        "What parasite prevention guidance is given for cats?",
        "What nutrition guidance is provided for kitten vs adult life stages?",
    ]

    fw_retriever = _build_retriever(chunks, _fw_embeddings())
    oa_retriever = _build_retriever(chunks, _oa_embeddings())

    # References are generated from OpenAI retriever contexts for consistency.
    oa_contexts = [[d.page_content for d in oa_retriever.invoke(q)] for q in questions]
    references = _build_reference_answers(questions, oa_contexts)

    fw_run = _run_provider(
        name="fireworks",
        questions=questions,
        retriever=fw_retriever,
        llm=_fw_chat(),
        references=references,
    )
    oa_run = _run_provider(
        name="openai",
        questions=questions,
        retriever=oa_retriever,
        llm=_oa_chat(),
        references=references,
    )

    fw_eval = _evaluate_with_ragas(fw_run)
    oa_eval = _evaluate_with_ragas(oa_run)

    print("\n=== RAGAS RESULTS ===")
    print("Fireworks:", fw_eval)
    print("OpenAI   :", oa_eval)
    print("\nCheck LangSmith traces for token usage and cost per query.")
    print("Suggested filters: project/activity1-ragas-cost, tags=fireworks|openai")


if __name__ == "__main__":
    main()

