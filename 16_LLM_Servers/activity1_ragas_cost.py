"""Activity 1 runner: Fireworks vs OpenAI RAGAS + LangSmith cost comparison.

Usage:
  uv run python activity1_ragas_cost.py

Required env vars:
  - FIREWORKS_API_KEY
  - OPENAI_API_KEY
  - LANGSMITH_API_KEY (for tracing/cost dashboard)

Optional env vars:
  - FIREWORKS_CHAT_MODEL (default: accounts/fireworks/models/gpt-oss-20b)
  - FIREWORKS_EMBEDDING_MODEL (default: fireworks/qwen3-embedding-8b)
  - OPENAI_CHAT_MODEL (default: gpt-4.1-mini)
  - OPENAI_EMBEDDING_MODEL (default: text-embedding-3-large)
  - RAG_DATA_DIR (default: data)
  - TOP_K (default: 4)
  - DISABLE_SSL_VERIFY (set to "1" to disable SSL verification for corporate proxies)
  - FIREWORKS_RPS (default: 0.15)
  - FIREWORKS_MAX_RETRIES (default: 10)
  - FIREWORKS_POST_INDEX_SLEEP_SECONDS (default: 30)
"""

from __future__ import annotations

import os
import random
import ssl
import time
from dataclasses import dataclass
from typing import Iterable

import httpx
import tiktoken
import urllib3
from dotenv import load_dotenv
from openai import RateLimitError
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


_FW_RATE_LIMITER: InMemoryRateLimiter | None = None


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_http_client() -> httpx.Client | None:
    """Return HTTP client with SSL verification disabled if needed for corporate proxy."""
    should_disable = (
        os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("https_proxy")
    )

    if should_disable:
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        return httpx.Client(
            verify=False,
            timeout=60.0,
            follow_redirects=True,
        )
    return None


def _get_async_http_client() -> httpx.AsyncClient | None:
    """Return async HTTP client with SSL verification disabled if needed for corporate proxy."""
    should_disable = (
        os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("https_proxy")
    )

    if should_disable:
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        return httpx.AsyncClient(
            verify=False,
            timeout=60.0,
            follow_redirects=True,
        )
    return None


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


def _fw_rate_limiter() -> InMemoryRateLimiter:
    global _FW_RATE_LIMITER
    if _FW_RATE_LIMITER is None:
        _FW_RATE_LIMITER = InMemoryRateLimiter(
            requests_per_second=float(os.environ.get("FIREWORKS_RPS", "0.15")),
            check_every_n_seconds=0.1,
            max_bucket_size=1,
        )
    return _FW_RATE_LIMITER


def _with_backoff(fn, *, label: str, attempts: int | None = None):
    max_attempts = attempts or int(os.environ.get("FIREWORKS_MAX_RETRIES", "10"))
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except RateLimitError:
            if attempt == max_attempts:
                raise
            sleep_s = min(60.0, (2 ** (attempt - 1)) + random.random())
            print(
                f"! {label}: rate limited "
                f"(attempt {attempt}/{max_attempts}), sleeping {sleep_s:.1f}s..."
            )
            time.sleep(sleep_s)


def _fw_embeddings() -> OpenAIEmbeddings:
    http_client = _get_http_client()
    return OpenAIEmbeddings(
        model=os.environ.get(
            "FIREWORKS_EMBEDDING_MODEL",
            "fireworks/qwen3-embedding-8b",
        ),
        openai_api_key=_require_env("FIREWORKS_API_KEY"),
        openai_api_base="https://api.fireworks.ai/inference/v1",
        check_embedding_ctx_length=False,
        dimensions=4096,
        http_client=http_client,
        max_retries=int(os.environ.get("FIREWORKS_MAX_RETRIES", "10")),
        request_timeout=60,
    )


def _oa_embeddings() -> OpenAIEmbeddings:
    http_client = _get_http_client()
    return OpenAIEmbeddings(
        model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        http_client=http_client,
        max_retries=10,
        request_timeout=60,
    )


def _fw_chat() -> ChatOpenAI:
    http_client = _get_http_client()
    async_http_client = _get_async_http_client()
    return ChatOpenAI(
        model=os.environ.get(
            "FIREWORKS_CHAT_MODEL",
            "accounts/fireworks/models/gpt-oss-20b",
        ),
        openai_api_key=_require_env("FIREWORKS_API_KEY"),
        openai_api_base="https://api.fireworks.ai/inference/v1",
        temperature=0,
        http_client=http_client,
        http_async_client=async_http_client,
        request_timeout=60,
        max_retries=int(os.environ.get("FIREWORKS_MAX_RETRIES", "10")),
        rate_limiter=_fw_rate_limiter(),
        tags=["fireworks"],
    )


def _oa_chat() -> ChatOpenAI:
    http_client = _get_http_client()
    async_http_client = _get_async_http_client()
    return ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
        http_client=http_client,
        http_async_client=async_http_client,
        request_timeout=60,
        max_retries=10,
        tags=["openai"],
    )


def _build_retriever(chunks: list[Document], embeddings: OpenAIEmbeddings):
    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="activity1",
    )
    return vectorstore.as_retriever(
        search_kwargs={"k": int(os.environ.get("TOP_K", "4"))}
    )


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
    return _with_backoff(
        lambda: llm.invoke(prompt).content,
        label="Fireworks chat completion" if "fireworks" in getattr(llm, "tags", []) else "chat completion",
    )


def _build_reference_answers(questions: list[str], contexts: list[list[str]]) -> list[str]:
    # Use a strong model once to generate consistent references from retrieved context.
    # We use a separate client instance without the "openai" tag so it doesn't skew the pipeline cost comparison
    http_client = _get_http_client()
    judge = ChatOpenAI(
        model=os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        openai_api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
        http_client=http_client,
        tags=["reference-builder"],
    )
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

        if name == "fireworks":
            answers.append(
                _with_backoff(
                    lambda q=q, ctx=ctx: _answer_question(llm, q, ctx),
                    label=f"Fireworks question: {q[:40]}...",
                )
            )
        else:
            answers.append(_answer_question(llm, q, ctx))

    return ProviderRun(
        name=name,
        questions=questions,
        answers=answers,
        contexts=contexts,
        references=references,
    )


def _evaluate_with_ragas(run: ProviderRun):
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

    http_client = _get_http_client()
    eval_llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=_require_env("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=8192,
        http_client=http_client,
        tags=["ragas-evaluation"],
    )

    try:
        from ragas.llms import LangchainLLMWrapper
        ragas_llm = LangchainLLMWrapper(eval_llm)
        result = evaluate(
            ds,
            metrics=[context_precision, faithfulness, answer_correctness],
            llm=ragas_llm,
        )
    except ImportError:
        result = evaluate(ds, metrics=[context_precision, faithfulness, answer_correctness])
    except Exception as e:
        print(f"! Warning: evaluation with custom LLM failed: {e}")
        print("   Falling back to context_precision and faithfulness only...")
        result = evaluate(ds, metrics=[context_precision, faithfulness])

    return result


def main() -> None:
    load_dotenv()

    should_disable_ssl = (
        os.environ.get("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("https_proxy")
    )

    if should_disable_ssl:
        os.environ["DISABLE_SSL_VERIFY"] = "1"
        os.environ["PYTHONHTTPSVERIFY"] = "0"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        ssl._create_default_https_context = ssl._create_unverified_context
        print("! SSL verification disabled due to proxy detection or DISABLE_SSL_VERIFY=1")

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault(
        "LANGCHAIN_PROJECT",
        os.environ.get("LANGCHAIN_PROJECT", "activity1-ragas-cost"),
    )

    _require_env("LANGSMITH_API_KEY")
    _require_env("FIREWORKS_API_KEY")
    _require_env("OPENAI_API_KEY")

    print("LangSmith tracing enabled:")
    print(f"   Project: {os.environ.get('LANGCHAIN_PROJECT')}")
    print("   All LLM calls will be traced with provider tags")

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

    print("Building Fireworks retriever...")
    fw_retriever = _with_backoff(
        lambda: _build_retriever(chunks, _fw_embeddings()),
        label="Fireworks embeddings/indexing",
    )

    cooldown = float(os.environ.get("FIREWORKS_POST_INDEX_SLEEP_SECONDS", "30"))
    if cooldown > 0:
        print(f"Cooling down for {cooldown:.0f}s after Fireworks indexing...")
        time.sleep(cooldown)

    print("Building OpenAI retriever...")
    oa_retriever = _build_retriever(chunks, _oa_embeddings())

    oa_contexts = [[d.page_content for d in oa_retriever.invoke(q)] for q in questions]
    references = _build_reference_answers(questions, oa_contexts)

    print("Running Fireworks QA...")
    fw_run = _run_provider(
        name="fireworks",
        questions=questions,
        retriever=fw_retriever,
        llm=_fw_chat(),
        references=references,
    )

    print("Running OpenAI QA...")
    oa_run = _run_provider(
        name="openai",
        questions=questions,
        retriever=oa_retriever,
        llm=_oa_chat(),
        references=references,
    )

    print("Evaluating Fireworks with RAGAS...")
    fw_eval = _evaluate_with_ragas(fw_run)

    print("Evaluating OpenAI with RAGAS...")
    oa_eval = _evaluate_with_ragas(oa_run)

    print("\n=== RAGAS RESULTS ===")

    print("\nFireworks AI:")
    if hasattr(fw_eval, "to_pandas"):
        df_fw = fw_eval.to_pandas()
        print(f"  Context Precision: {df_fw['context_precision'].mean():.4f}")
        print(f"  Faithfulness: {df_fw['faithfulness'].mean():.4f}")
        if "answer_correctness" in df_fw.columns and not df_fw["answer_correctness"].isna().all():
            print(f"  Answer Correctness: {df_fw['answer_correctness'].mean():.4f}")
        else:
            print("  Answer Correctness: N/A (evaluation incomplete)")
    else:
        print(f"  {fw_eval}")

    print("\nOpenAI:")
    if hasattr(oa_eval, "to_pandas"):
        df_oa = oa_eval.to_pandas()
        print(f"  Context Precision: {df_oa['context_precision'].mean():.4f}")
        print(f"  Faithfulness: {df_oa['faithfulness'].mean():.4f}")
        if "answer_correctness" in df_oa.columns and not df_oa["answer_correctness"].isna().all():
            print(f"  Answer Correctness: {df_oa['answer_correctness'].mean():.4f}")
        else:
            print("  Answer Correctness: N/A (evaluation incomplete)")
    else:
        print(f"  {oa_eval}")


if __name__ == "__main__":
    main()