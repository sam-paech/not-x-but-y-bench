#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Creates a leaderboard chart from results JSON files in a directory.
Extracts model names and their overall not-x-but-y scores (rate_per_1k).
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def extract_scores_from_results(results_dir: Path) -> list:
    """
    Iterate over all JSON files in results_dir and extract model scores.

    Returns:
        List of tuples: (model_name, rate_per_1k, total_hits, total_chars)
    """
    scores = []

    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Each JSON file contains one or more models
            for model_id, model_data in data.items():
                if isinstance(model_data, dict) and "summary" in model_data:
                    summary = model_data["summary"]
                    model_name = model_data.get("test_model", model_id)
                    rate = summary.get("rate_per_1k", 0.0)
                    hits = summary.get("total_hits", 0)
                    chars = summary.get("total_chars", 0)

                    scores.append((model_name, rate, hits, chars))
        except Exception as e:
            print(f"Warning: Could not process {json_file}: {e}")

    return scores


def create_leaderboard(scores: list, output_path: Path) -> None:
    """
    Create a dark-mode leaderboard chart with gradient colors.

    Args:
        scores: List of tuples (model_name, rate_per_1k, total_hits, total_chars)
        output_path: Path where to save the chart image
    """
    # Add human baseline (computed from human_writing_samples/*.txt)
    scores_with_human = scores + [("Human baseline", 0.065, np.nan, np.nan)]

    df = (pd.DataFrame(scores_with_human, columns=["model", "rate", "hits", "chars"])
            .sort_values("rate", ascending=False)
            .drop_duplicates(subset="model", keep="first")  # keep highest-rate run
            .reset_index(drop=True))

    num_bars = len(df)

    # --------------------------------------------------------------------------- #
    # Styling and colour setup                                                   #
    # --------------------------------------------------------------------------- #
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Segoe UI', 'DejaVu Sans']

    bot_color = "#00421d"
    mid_color = "#005f27"
    top_color = "#39FF14"

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_green_gradient", [bot_color, mid_color, top_color]
    )
    gradient_colors = [custom_cmap(i) for i in np.linspace(0, 1, num_bars)][::-1]

    # --------------------------------------------------------------------------- #
    # Chart creation                                                             #
    # --------------------------------------------------------------------------- #
    fig_height = max(3, num_bars * 0.20)  # compact layout
    fig, ax = plt.subplots(figsize=(10, fig_height))

    sns.barplot(
        x='rate',
        y='model',
        data=df,
        hue='model',
        palette=gradient_colors,
        legend=False,
        ax=ax
    )

    # --------------------------------------------------------------------------- #
    # Annotation and labelling                                                   #
    # --------------------------------------------------------------------------- #
    for patch in ax.patches:
        width = patch.get_width()
        ax.text(
            width + (ax.get_xlim()[1] * 0.01),
            patch.get_y() + patch.get_height() / 2,
            f'{width:.3f}',  # 3-dp precision for clarity
            va='center',
            ha='left',
            fontsize=9,
            color='lightgray'
        )

    ax.set_title(
        '"Not x, but y" Slop Leaderboard\n'
        r'$\it{phrases\ per\ 1{,}000\ characters}$',
        fontsize=16,
        fontweight='bold',
        pad=10
    )

    ax.set_xlabel('Not-X-but-Y phrases per 1,000 Characters', fontsize=10)
    ax.set_ylabel('')
    ax.tick_params(axis='y', labelsize=9)

    # --------------------------------------------------------------------------- #
    # Final tweaks                                                               #
    # --------------------------------------------------------------------------- #
    ax.set_xlim(right=df['rate'].max() * 1.18)  # padding for labels
    sns.despine(left=True, bottom=False)
    plt.tight_layout()

    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart saved to {output_path}")

    # Also print summary to console
    print(f"\nLeaderboard Summary:")
    print(f"{'Model':<40} {'Rate per 1k':<12}")
    print("=" * 52)
    for _, row in df.iterrows():
        print(f"{row['model']:<40} {row['rate']:<12.3f}")


def main():
    ap = argparse.ArgumentParser(
        description="Create a leaderboard chart from results JSON files"
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default="results/",
        help="Directory containing results JSON files (default: results/)"
    )
    ap.add_argument(
        "--output",
        type=str,
        default="leaderboard.png",
        help="Output image file path (default: leaderboard.png)"
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    if not results_dir.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return 1

    print(f"Scanning {results_dir} for results...")
    scores = extract_scores_from_results(results_dir)

    if not scores:
        print("Error: No valid results found in the directory")
        return 1

    print(f"Found {len(scores)} model results")
    create_leaderboard(scores, output_path)

    return 0


if __name__ == "__main__":
    exit(main())
