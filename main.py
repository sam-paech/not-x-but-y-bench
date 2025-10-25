# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from src.config import load_env, get_api_config
from src.api import ApiClient
from src.scorer import score_text
from src.io_utils import atomic_update_model_results, ensure_model_header

try:
    from tqdm import tqdm
except Exception:  # fallback
    def tqdm(it, **kwargs):
        return it


def _prompt_text(user_prompt: str) -> str:
    return (
        "Write approximately 1,000 words on the following writing prompt. "
        "Do not use tables.\n\n"
        f"Prompt: {user_prompt}"
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run longform eval and compute not-x-but-y rate.")
    ap.add_argument("model", type=str, help="Model ID to query (e.g., gpt-4o, meta-llama/... etc.)")
    ap.add_argument("--prompts", type=str, default="prompts.json", help="Path to prompts.json (list[str])")
    ap.add_argument("--results", type=str, default="results.json", help="Path to results JSON (append-only by model)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel threads")
    ap.add_argument("--timeout", type=float, default=480.0, help="HTTP timeout seconds")
    ap.add_argument("--max-tokens", type=int, default=8096, help="Max tokens per generation")
    ap.add_argument("--n-prompts", type=int, default=300, help="Number of prompts to use")
    ap.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    ap.add_argument("--retry-delay", type=float, default=5.0, help="Delay between retries in seconds")
    args = ap.parse_args()

    load_env()
    base_url, api_key = get_api_config()

    model_id = args.model
    prompts_path = Path(args.prompts)
    results_path = Path(args.results)

    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("prompts.json must be a JSON list of strings")

    # Limit to n_prompts
    prompts = prompts[:args.n_prompts]

    client = ApiClient(
        base_url=base_url,
        api_key=api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    # Write model header if missing and mark start time
    ensure_model_header(
        results_path=results_path,
        model_id=model_id,
        endpoint=base_url,
        params={
            "temperature": 0.7,
            "min_p": 0.1,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "prompts_path": str(prompts_path),
        },
        started_at=_now_iso(),
    )

    def _work(idx_prompt_tuple):
        idx, prompt = idx_prompt_tuple
        try:
            text = client.generate(
                model=model_id,
                prompt_text=_prompt_text(prompt),
                max_tokens=args.max_tokens,
            )
            s = score_text(text)
            sample = {
                "prompt_index": idx,
                "prompt": prompt,
                "output": text,
                "chars": s["chars"],
                "hits": s["hits"],
                "rate_per_1k": s["rate_per_1k"],
            }
            # Progressive, atomic, thread-safe update
            atomic_update_model_results(results_path, model_id, sample)
            return sample
        except Exception as e:
            # Record the error for this specific prompt
            sample = {
                "prompt_index": idx,
                "prompt": prompt,
                "output": "",
                "chars": 0,
                "hits": 0,
                "rate_per_1k": 0.0,
                "error": str(e),
            }
            atomic_update_model_results(results_path, model_id, sample)
            return sample

    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, p in enumerate(prompts):
            futures.append(ex.submit(_work, (i, p)))

        total_chars = 0
        total_hits = 0
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating + Scoring"):
            res = fut.result()
            total_chars += res["chars"]
            total_hits += res["hits"]

    # write completion timestamp and final summary
    atomic_update_model_results(
        results_path,
        model_id,
        None,  # no new sample
        completed_at=_now_iso()
    )

    overall_rate = (total_hits * 1000.0 / total_chars) if total_chars > 0 else 0.0
    print("\nSUMMARY")
    print(f"model={model_id}")
    print(f"chars={total_chars}  hits={total_hits}  rate_per_1k={overall_rate:.3f}")
    print(f"results_file={results_path}")


if __name__ == "__main__":
    main()
