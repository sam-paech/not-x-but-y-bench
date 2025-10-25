#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Computes the human baseline "not x, but y" score from text files
in the human_writing_samples directory.
"""

import argparse
from pathlib import Path
from src.scorer import score_text

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


def compute_human_baseline(samples_dir: Path, verbose: bool = False) -> dict:
    """
    Compute the aggregate "not x, but y" score across all text files.

    Args:
        samples_dir: Directory containing .txt files
        verbose: If True, print per-file statistics

    Returns:
        Dictionary with total_chars, total_hits, and rate_per_1k
    """
    text_files = list(samples_dir.glob("*.txt"))

    if not text_files:
        raise ValueError(f"No .txt files found in {samples_dir}")

    total_chars = 0
    total_hits = 0
    file_results = []

    for txt_file in tqdm(text_files, desc="Processing samples"):
        try:
            text = txt_file.read_text(encoding='utf-8')
            score = score_text(text)

            total_chars += score["chars"]
            total_hits += score["hits"]

            file_results.append({
                "file": txt_file.name,
                "chars": score["chars"],
                "hits": score["hits"],
                "rate_per_1k": score["rate_per_1k"],
            })
        except Exception as e:
            print(f"Warning: Could not process {txt_file.name}: {e}")

    overall_rate = (total_hits * 1000.0 / total_chars) if total_chars > 0 else 0.0

    if verbose:
        print("\n" + "=" * 80)
        print("PER-FILE BREAKDOWN")
        print("=" * 80)
        print(f"{'File':<60} {'Chars':<12} {'Hits':<8} {'Rate/1k':<10}")
        print("-" * 80)
        for result in sorted(file_results, key=lambda x: x["rate_per_1k"], reverse=True):
            print(f"{result['file']:<60} {result['chars']:<12,} {result['hits']:<8} {result['rate_per_1k']:<10.3f}")
        print("=" * 80)

    return {
        "total_files": len(file_results),
        "total_chars": total_chars,
        "total_hits": total_hits,
        "rate_per_1k": overall_rate,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compute human baseline 'not x, but y' score"
    )
    ap.add_argument(
        "--samples-dir",
        type=str,
        default="human_writing_samples/",
        help="Directory containing human writing samples (default: human_writing_samples/)"
    )
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-file statistics"
    )
    args = ap.parse_args()

    samples_dir = Path(args.samples_dir)

    if not samples_dir.exists():
        print(f"Error: Samples directory '{samples_dir}' does not exist")
        return 1

    print(f"Computing human baseline from {samples_dir}...")

    result = compute_human_baseline(samples_dir, verbose=args.verbose)

    print("\n" + "=" * 80)
    print("HUMAN BASELINE SUMMARY")
    print("=" * 80)
    print(f"Total files processed:  {result['total_files']}")
    print(f"Total characters:       {result['total_chars']:,}")
    print(f"Total hits:             {result['total_hits']}")
    print(f"Rate per 1,000 chars:   {result['rate_per_1k']:.3f}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
