# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
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


def _score_sample(sample_dict):
    """
    Worker function for multiprocessing Pool.
    Takes a sample dict with 'output' key and returns it with scoring added.
    """
    text = sample_dict.get("output", "")
    s = score_text(text)
    sample_dict["chars"] = s["chars"]
    sample_dict["hits"] = s["hits"]
    sample_dict["rate_per_1k"] = s["rate_per_1k"]
    return sample_dict


def _get_completed_prompts(results_path: Path, model_id: str) -> dict:
    """
    Load existing results and return a dict of completed samples by prompt_index.
    Returns dict with:
      - 'generated': set of prompt_indices that have been generated (output exists)
      - 'samples': dict mapping prompt_index -> sample
    """
    if not results_path.exists():
        return {'generated': set(), 'samples': {}}

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        model_data = data.get(model_id, {})
        samples = model_data.get("samples", [])

        generated = set()
        samples_by_idx = {}

        for sample in samples:
            idx = sample.get("prompt_index")
            if idx is None:
                continue

            samples_by_idx[idx] = sample

            # Has output and no error = generated
            if sample.get("output") and not sample.get("error"):
                generated.add(idx)

        return {'generated': generated, 'samples': samples_by_idx}
    except Exception:
        return {'generated': set(), 'samples': {}}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run longform eval and compute not-x-but-y rate.")
    ap.add_argument("model", type=str, help="Model ID to query (e.g., gpt-4o, meta-llama/... etc.)")
    ap.add_argument("--prompts", type=str, default="prompts.json", help="Path to prompts.json (list[str])")
    ap.add_argument("--results", type=str, default="results.json", help="Path to results JSON (append-only by model)")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel threads")
    ap.add_argument("--timeout", type=float, default=480.0, help="HTTP timeout seconds")
    ap.add_argument("--max-tokens", type=int, default=8096, help="Max tokens per generation")
    ap.add_argument("--n-prompts", type=int, default=300, help="Number of prompts to use")
    ap.add_argument("--max-retries", type=int, default=3, help="Max retries per request")
    ap.add_argument("--retry-delay", type=float, default=5.0, help="Delay between retries in seconds")
    ap.add_argument("--scoring-workers", type=int, default=12, help="Number of parallel processes for scoring (default: 12)")
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

    # Check for existing results to resume from
    completed = _get_completed_prompts(results_path, model_id)
    already_generated = completed['generated']
    existing_samples = completed['samples']

    # Report resume status
    if already_generated:
        print(f"Found existing results: {len(already_generated)} already generated")
        print(f"Resuming from checkpoint (will re-score all samples)...")

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

    # Phase 1: Generate text outputs (threaded for I/O concurrency)
    # Skip prompts that are already generated
    prompts_to_generate = [(i, p) for i, p in enumerate(prompts) if i not in already_generated]

    def _generate(idx_prompt_tuple):
        idx, prompt = idx_prompt_tuple
        try:
            text = client.generate(
                model=model_id,
                prompt_text=_prompt_text(prompt),
                max_tokens=args.max_tokens,
            )
            sample = {
                "prompt_index": idx,
                "prompt": prompt,
                "output": text,
                # Don't set chars/hits/rate_per_1k here - they'll be added in Phase 2
            }
        except Exception as e:
            sample = {
                "prompt_index": idx,
                "prompt": prompt,
                "output": "",
                "error": str(e),
                # Don't set chars/hits/rate_per_1k for errors either
            }

        # Progressive, atomic, thread-safe save
        atomic_update_model_results(results_path, model_id, sample)
        return sample

    if prompts_to_generate:
        print(f"Phase 1: Generating {len(prompts_to_generate)} outputs with {args.workers} worker threads...")
        print(f"  (Skipping {len(already_generated)} already generated)")
        generated_samples = []
        futures = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for idx_prompt in prompts_to_generate:
                futures.append(ex.submit(_generate, idx_prompt))

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
                generated_samples.append(fut.result())
    else:
        print(f"Phase 1: All {len(prompts)} prompts already generated, skipping generation phase")
        generated_samples = []

    # Combine newly generated with existing samples
    all_samples_dict = existing_samples.copy()
    for sample in generated_samples:
        all_samples_dict[sample['prompt_index']] = sample

    # Convert to list for Phase 2, sorted by prompt_index
    all_samples = [all_samples_dict[i] for i in sorted(all_samples_dict.keys())]

    # Phase 2: Score all outputs (multiprocessing for CPU-bound POS tagging)
    # Always score all samples to ensure consistency and allow for scoring updates
    if all_samples:
        print(f"\nPhase 2: Scoring {len(all_samples)} outputs with {args.scoring_workers} worker processes...")

        with Pool(processes=args.scoring_workers) as pool:
            for scored_sample in tqdm(
                pool.imap(_score_sample, all_samples),
                total=len(all_samples),
                desc="Scoring"
            ):
                # Progressive, atomic save with updated scores
                atomic_update_model_results(results_path, model_id, scored_sample)
    else:
        print(f"\nPhase 2: No samples to score")

    # Calculate final totals from all samples (including already-scored ones)
    total_chars = 0
    total_hits = 0
    for idx in sorted(all_samples_dict.keys()):
        sample = all_samples_dict[idx]
        total_chars += sample.get("chars", 0)
        total_hits += sample.get("hits", 0)

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
