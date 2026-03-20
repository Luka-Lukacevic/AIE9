"""Helper script to aggregate LangSmith cost data by provider.

LangSmith only knows OpenAI pricing natively. Fireworks calls show $0 cost
because LangSmith doesn't have Fireworks pricing tables.

This script:
- Gets token counts for both providers from LangSmith traces
- Uses OpenAI's reported cost directly
- Estimates Fireworks cost from token counts using published pricing

Usage:
  uv run python langsmith_cost_summary.py

Required env vars:
  - LANGSMITH_API_KEY
  - LANGCHAIN_PROJECT (default: llm_servers)
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    from langsmith import Client
except ImportError:
    print("Missing langsmith package. Install with: uv add langsmith")
    exit(1)

# Fireworks pricing (per 1M tokens) —  fireworks.ai/pricing
# gpt-oss-20b serverless pricing
FW_CHAT_INPUT_PER_1M = 0.07   # $/1M input tokens
FW_CHAT_OUTPUT_PER_1M = 0.30  # $/1M output tokens
FW_EMBED_PER_1M = 0.016       # $/1M tokens for embeddings


def get_cost_summary(project_name: str = "llm_servers", days: int = 7):
    """Get cost summary from LangSmith for the specified project."""
    load_dotenv()

    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("Missing LANGSMITH_API_KEY environment variable")
        return

    endpoint = os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT")
    client = Client(api_key=api_key, api_url=endpoint) if endpoint else Client(api_key=api_key)

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    print(f"Fetching cost data for project: {project_name}")
    print(f"Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}\n")

    try:
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
        ))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        return

    if not runs:
        print("No runs found.")
        return

    # Buckets
    fw = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "runs": 0}
    oa = {"cost": 0.0, "input_tokens": 0, "output_tokens": 0, "runs": 0}
    skipped_eval = 0

    for run in runs:
        # Only count ChatOpenAI runs (the actual QA calls), ignore the parent chain runs
        if run.run_type != "llm":
            continue

        tags = run.tags or []
        name = (run.name or "").lower()
        
        # We only want to count the actual pipeline QA calls.
        # RAGAS evaluation runs and reference-builder runs should be excluded.
        if "ragas-evaluation" in tags or "gpt-4o-mini" in name:
            skipped_eval += 1
            continue

        if "reference-builder" in tags:
            skipped_eval += 1
            continue
            
        # Determine provider from tags or model name
        is_fw = "fireworks" in tags or "fireworks" in name
        is_oa = "openai" in tags or ("gpt-4" in name and "fireworks" not in name)

        # Also skip runs that are neither fw nor oa (like plain vectorstore traces)
        if not is_fw and not is_oa:
            continue
            
        # Only count ChatOpenAI runs (the actual QA calls), ignore the parent chain runs
        if run.run_type != "llm":
            continue

        # Token counts from run
        prompt_tokens = 0
        completion_tokens = 0
        if run.prompt_tokens:
            prompt_tokens = run.prompt_tokens
        if run.completion_tokens:
            completion_tokens = run.completion_tokens

        # LangSmith-reported cost (only works for OpenAI)
        run_cost = 0.0
        if hasattr(run, "total_cost") and run.total_cost:
            run_cost = float(run.total_cost)

        if is_fw:
            fw["input_tokens"] += prompt_tokens
            fw["output_tokens"] += completion_tokens
            fw["runs"] += 1
            # Estimate cost from tokens
            fw["cost"] += (
                prompt_tokens * FW_CHAT_INPUT_PER_1M / 1_000_000
                + completion_tokens * FW_CHAT_OUTPUT_PER_1M / 1_000_000
            )
        elif is_oa:
            oa["input_tokens"] += prompt_tokens
            oa["output_tokens"] += completion_tokens
            oa["runs"] += 1
            oa["cost"] += run_cost  # Use LangSmith's known OpenAI pricing

    # Print results
    print("=" * 60)
    print("COST SUMMARY BY PROVIDER (Pipeline only, RAGAS judge excluded)")
    print("=" * 60)
    if skipped_eval > 0:
        print(f"Skipped {skipped_eval} RAGAS evaluation trace(s)\n")

    print(f"\nFireworks AI (gpt-oss-20b):")
    print(f"  Runs:           {fw['runs']}")
    print(f"  Input tokens:   {fw['input_tokens']:,}")
    print(f"  Output tokens:  {fw['output_tokens']:,}")
    print(f"  Total tokens:   {fw['input_tokens'] + fw['output_tokens']:,}")
    print(f"  Total cost:      ${fw['cost']:.6f}")
    if fw["runs"] > 0:
        print(f"  Avg cost/run:   ${fw['cost'] / fw['runs']:.6f}")

    print(f"\nOpenAI (gpt-4.1-mini):")
    print(f"  Runs:           {oa['runs']}")
    print(f"  Input tokens:   {oa['input_tokens']:,}")
    print(f"  Output tokens:  {oa['output_tokens']:,}")
    print(f"  Total tokens:   {oa['input_tokens'] + oa['output_tokens']:,}")
    print(f"  Total cost:     ${oa['cost']:.6f}  (from LangSmith)")
    if oa["runs"] > 0:
        print(f"  Avg cost/run:   ${oa['cost'] / oa['runs']:.6f}")

    print(f"\nComparison:")
    if fw["cost"] > 0 and oa["cost"] > 0:
        ratio = oa["cost"] / fw["cost"]
        cheaper = "Fireworks" if ratio > 1 else "OpenAI"
        print(f"  OpenAI/Fireworks cost ratio: {ratio:.2f}x")
        print(f"  {cheaper} is cheaper for this workload")
    elif fw["runs"] == 0:
        print("  No Fireworks runs found with token data.")
        print("  Note: LangSmith may not track Fireworks token counts")
        print("  because Fireworks uses a non-standard API base URL.")
    print()


if __name__ == "__main__":
    project = os.environ.get("LANGCHAIN_PROJECT", "llm_servers")
    get_cost_summary(project_name=project, days=7)
