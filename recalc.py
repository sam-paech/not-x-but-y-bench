#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recalculate all scores in results JSON files.

This script iterates over results files and recalculates:
- rate_per_1k for each sample's output
- summary statistics (total_chars, total_hits, rate_per_1k)

Useful when the scoring logic has been updated and you need to
recompute scores for existing outputs without re-running evaluations.

Usage:
  python recalc.py --results-dir results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from src.scorer import score_text

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


def recalculate_file(results_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    Recalculate scores for all samples in a results file.

    Args:
        results_path: Path to the results JSON file
        dry_run: If True, don't write changes to disk

    Returns:
        Dictionary with statistics about the recalculation
    """
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return {
            "file": results_path.name,
            "success": False,
            "error": str(e),
        }

    stats = {
        "file": results_path.name,
        "success": True,
        "models_processed": 0,
        "samples_recalculated": 0,
        "samples_with_errors": 0,
        "changes": [],
    }

    # Process each model in the file
    for model_id, model_data in data.items():
        if not isinstance(model_data, dict):
            continue

        if "samples" not in model_data:
            continue

        stats["models_processed"] += 1

        total_chars = 0
        total_hits = 0
        error_count = 0

        # Recalculate each sample
        for sample in model_data["samples"]:
            if "output" not in sample:
                continue

            # Skip samples that already have errors (failed generations)
            if "error" in sample and sample["error"]:
                error_count += 1
                continue

            output_text = sample["output"]

            # Recalculate score
            old_rate = sample.get("rate_per_1k", 0.0)
            old_hits = sample.get("hits", 0)
            old_chars = sample.get("chars", 0)

            new_score = score_text(output_text)

            sample["chars"] = new_score["chars"]
            sample["hits"] = new_score["hits"]
            sample["rate_per_1k"] = new_score["rate_per_1k"]

            total_chars += new_score["chars"]
            total_hits += new_score["hits"]

            stats["samples_recalculated"] += 1

            # Track significant changes (>5% difference in rate)
            if old_rate > 0 and abs(new_score["rate_per_1k"] - old_rate) / old_rate > 0.05:
                stats["changes"].append({
                    "model": model_id,
                    "prompt_index": sample.get("prompt_index"),
                    "old_rate": old_rate,
                    "new_rate": new_score["rate_per_1k"],
                    "change_pct": ((new_score["rate_per_1k"] - old_rate) / old_rate * 100),
                })

        stats["samples_with_errors"] = error_count

        # Update summary
        overall_rate = (total_hits * 1000.0 / total_chars) if total_chars > 0 else 0.0

        if "summary" not in model_data:
            model_data["summary"] = {}

        old_summary_rate = model_data["summary"].get("rate_per_1k", 0.0)

        model_data["summary"]["total_prompts"] = stats["samples_recalculated"] + error_count
        model_data["summary"]["total_chars"] = total_chars
        model_data["summary"]["total_hits"] = total_hits
        model_data["summary"]["rate_per_1k"] = overall_rate

        # Track summary changes
        if old_summary_rate > 0 and abs(overall_rate - old_summary_rate) / old_summary_rate > 0.01:
            print(f"  {model_id}: {old_summary_rate:.3f} -> {overall_rate:.3f} "
                  f"({((overall_rate - old_summary_rate) / old_summary_rate * 100):+.1f}%)")

    # Write updated data back to file
    if not dry_run:
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            stats["success"] = False
            stats["error"] = f"Failed to write file: {e}"

    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Recalculate scores in results JSON files"
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Directory containing results JSON files (default: results/)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write changes to disk, just show what would change"
    )
    ap.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed statistics"
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return 1

    json_files = list(results_dir.glob("*.json"))

    if not json_files:
        print(f"Error: No JSON files found in '{results_dir}'")
        return 1

    print(f"{'DRY RUN: ' if args.dry_run else ''}Recalculating scores in {len(json_files)} files...")
    if args.dry_run:
        print("(No files will be modified)")

    all_stats = []
    total_samples = 0
    total_errors = 0
    total_models = 0

    for json_file in tqdm(json_files, desc="Processing files"):
        stats = recalculate_file(json_file, dry_run=args.dry_run)
        all_stats.append(stats)

        if stats["success"]:
            total_samples += stats["samples_recalculated"]
            total_errors += stats["samples_with_errors"]
            total_models += stats["models_processed"]

    print("\n" + "=" * 80)
    print("RECALCULATION SUMMARY")
    print("=" * 80)
    print(f"Files processed:        {len([s for s in all_stats if s['success']])}/{len(json_files)}")
    print(f"Models processed:       {total_models}")
    print(f"Samples recalculated:   {total_samples}")
    print(f"Samples with errors:    {total_errors}")

    if args.verbose and any(s.get("changes") for s in all_stats):
        print("\n" + "=" * 80)
        print("SIGNIFICANT CHANGES (>5% difference)")
        print("=" * 80)
        for stats in all_stats:
            if stats.get("changes"):
                print(f"\n{stats['file']}:")
                for change in stats["changes"][:5]:  # Show first 5
                    print(f"  Prompt {change['prompt_index']}: "
                          f"{change['old_rate']:.3f} -> {change['new_rate']:.3f} "
                          f"({change['change_pct']:+.1f}%)")
                if len(stats["changes"]) > 5:
                    print(f"  ... and {len(stats['changes']) - 5} more changes")

    # Show failed files
    failed = [s for s in all_stats if not s["success"]]
    if failed:
        print("\n" + "=" * 80)
        print("FAILED FILES")
        print("=" * 80)
        for stats in failed:
            print(f"{stats['file']}: {stats.get('error', 'Unknown error')}")

    print("=" * 80)

    if args.dry_run:
        print("\nDry run complete. No files were modified.")
        print("Run without --dry-run to apply changes.")

    return 0


if __name__ == "__main__":
    exit(main())
