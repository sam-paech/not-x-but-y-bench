#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute human-baseline "not X / Y" contrast rate from .txt files using a two-stage approach.

Definition:
  hits_per_1000 = total_unique_sentence_hits * 1000 / total_characters

Two-Stage Matching:
  • Stage 1: Surface-form regexes (V3 baseline) on normalized text
  • Stage 2: POS-based regexes on VERB-tagged text (only if Stage 1 didn't match)

Semantics:
  • Cross-sentence matches allowed. Compute covered sentence range [lo, hi] per match.
  • Merge overlapping ranges into one hit (e.g., A=2–3 and B=3–4 → 1 hit).
  • Files are scanned in overlapping chunks to keep regex ingestion bounded.
  • Dedup across chunks by char-interval overlap, not sentence-start.
  • Parallelized over files with ProcessPoolExecutor.
  • Progress bars for file enumeration and per-file completion.
  • Uniform random sampling over all hits using per-hit random priorities and a global reservoir.

Usage:
  python not-x-but-y-calc-human-baseline.py \
    --dir human_writing_samples \
    --workers 8 \
    --chunk-size 20000 \
    --overlap 500 \
    --sample-size 100 \
    --sample-seed 0
"""

from __future__ import annotations
import argparse
import os
import re
import sys
import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from bisect import bisect_left, bisect_right


# ---- optional progress -----------------------------------------------
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it  # graceful fallback: no progress bar

# ──────────────────────────────────────────────────────────────────────
# 1) Normalization and Two-Stage Regexes
# ──────────────────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize smart quotes and em-dashes to ASCII equivalents."""
    replacements = {
        chr(8220): '"',  # LEFT DOUBLE QUOTATION MARK
        chr(8221): '"',  # RIGHT DOUBLE QUOTATION MARK
        chr(8216): "'",  # LEFT SINGLE QUOTATION MARK
        chr(8217): "'",  # RIGHT SINGLE QUOTATION MARK
        chr(8212): '-',  # EM DASH
        chr(8211): '-',  # EN DASH
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Import the two-stage regex sets
try:
    import regexes_v3
    import regexes_pos
    HAS_REGEXES = True
except ImportError:
    HAS_REGEXES = False
    print("WARNING: Could not import regexes_v3 or regexes_pos. Using fallback patterns.", file=sys.stderr)

# Stage 1: V3 baseline surface-form patterns
STAGE1_REGEXES: Dict[str, re.Pattern] = {}
if HAS_REGEXES:
    STAGE1_REGEXES = regexes_v3.compiled
else:
    # Fallback to simplified patterns if imports fail
    MAXG = 160
    PRON = r"(?:it|they|this|that|you)"
    BE = r"(?:is|are|was|were)"
    BE_NEG = r"(?:is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|wasn't|weren't|ain't)"

    STAGE1_REGEXES = {
        "RE_NOT_BUT": re.compile(rf"""
            \b(?:(?:{BE_NEG})|not(?!\s+(?:that|only)\b))\s+
            (?:(?!\bbut\b|[.?!]).){{1,100}}?
            [,;:]\s*but\s+
            (?!when\b|while\b|which\b|who\b|whom\b|whose\b|where\b|if\b|that\b|as\b|because\b|although\b|though\b)
        """, re.I | re.X),
        "RE_NOT_DASH": re.compile(rf"""
            \b(?:\w+n't|not)\s+(?:just|only|merely)?\s+
            (?:(?![.?!]).){{1,160}}?
            (?:-|\s-\s|[\u2014\u2013])\s*
            {PRON}\s+(?:(?:'re|are|'s|is|were|was)\b|(?!'re|are|'s|is|were|was)[*_~]*[a-z]\w*)
        """, re.I | re.X),
    }

# Stage 2: POS-based patterns (only applied if Stage 1 doesn't match)
STAGE2_REGEXES: Dict[str, re.Pattern] = {}
if HAS_REGEXES:
    # Select most effective POS patterns from regexes_pos
    STAGE2_REGEXES = {
        "POS_DOESNT_VERB": regexes_pos.RE_POS_DOESNT_VERB,
        "POS_GERUND_FRAGMENT": regexes_pos.RE_POS_GERUND_FRAGMENT,
        "POS_DIALOGUE_ATTR": regexes_pos.RE_POS_DIALOGUE_ATTR,
        "POS_I_AM_NOT_SEMI": regexes_pos.RE_POS_I_AM_NOT_SEMI,
        "POS_NOT_ANYMORE_ITS": regexes_pos.RE_POS_NOT_ANYMORE_ITS,
    }

# Combined for backward compatibility
COMPILED_REGEXES = STAGE1_REGEXES

# POS tagging support (optional)
try:
    from pos_tagger import tag_stream_with_offsets
    HAS_POS_TAGGER = True
except ImportError:
    HAS_POS_TAGGER = False
    def tag_stream_with_offsets(text: str, pos_type: str = 'verb'):
        # Fallback: identity stream and one full-span piece map
        return text, [(0, len(text), 0, len(text))]


_SENT_SPLIT = re.compile(r'[^.!?]*[.!?]', flags=re.S)

def sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    last_end = 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((m.start(), m.end()))
        last_end = m.end()
    if last_end < len(text):
        spans.append((last_end, len(text)))  # trailing fragment
    return spans

def _covered_sentence_range(spans: Sequence[Tuple[int,int]], start: int, end: int) -> Tuple[int,int] | None:
    """
    Map [start,end) (chunk-local) to inclusive sentence index range [lo,hi].
    Returns None if no overlap.
    """
    if not spans or start >= end:
        return None
    starts = [s for s, _ in spans]
    ends   = [e for _, e in spans]
    lo = bisect_right(ends, start)  # first sentence with end > start
    hi = bisect_left(starts, end) - 1  # last sentence with start < end
    if lo >= len(spans) or hi < 0 or lo > hi:
        return None
    return lo, hi

def _merge_sentence_blocks(items: List[dict]) -> List[dict]:
    """
    items: [{'lo':int,'hi':int,'raw_start':int,'raw_end':int,'rule':str}]
    Merge overlapping [lo,hi] by sentence index. Keep first item's rule.
    """
    if not items:
        return []
    items_sorted = sorted(items, key=lambda d: (d['lo'], d['hi'], d['raw_start']))
    merged = []
    cur = items_sorted[0].copy()
    for it in items_sorted[1:]:
        if it['lo'] <= cur['hi']:  # overlaps at least one sentence
            cur['hi'] = max(cur['hi'], it['hi'])
            cur['raw_end'] = max(cur['raw_end'], it['raw_end'])
            # keep cur['rule'] from the first item
        else:
            merged.append(cur)
            cur = it.copy()
    merged.append(cur)
    return merged

def _ranges_overlap(ranges: List[Tuple[int,int]], s: int, e: int) -> bool:
    # simple O(n) overlap check; hit counts are small
    for a, b in ranges:
        if a < e and s < b:
            return True
    return False

def _add_range(ranges: List[Tuple[int,int]], s: int, e: int) -> None:
    ranges.append((s, e))
    ranges.sort()
    merged: List[Tuple[int,int]] = []
    for a, b in ranges:
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    ranges[:] = merged


# ──────────────────────────────────────────────────────────────────────
# 2) Chunked scanning with sentence-level uniqueness + sampling
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FileStats:
    file: str
    chars: int
    hits: int
    rate_per_1000: float

@dataclass
class SampleHit:
    rule: str
    text: str  # the matched sentence (trimmed)

def _sent_index(spans: Sequence[Tuple[int,int]], pos: int) -> int | None:
    lo, hi = 0, len(spans) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = spans[mid]
        if s <= pos < e:
            return mid
        if pos < s:
            hi = mid - 1
        else:
            lo = mid + 1
    return None


def _reservoir_maybe_add(heap: List[Tuple[float, SampleHit]], k: int, key: float, item: SampleHit) -> None:
    """
    Keep the k lowest keys using a max-heap encoded as a min-heap of negative keys.
    heap stores tuples (-key, item)
    """
    if k <= 0:
        return
    neg = -key
    if len(heap) < k:
        heapq.heappush(heap, (neg, item))
    else:
        # heap[0] has the smallest negative → largest positive key
        if neg > heap[0][0]:
            heapq.heapreplace(heap, (neg, item))

def _scan_chunk_for_hits(chunk: str,
                         abs_offset: int,
                         covered_ranges: List[Tuple[int,int]],
                         rng: random.Random,
                         heap: List[Tuple[float, SampleHit]],
                         k: int) -> int:
    """
    Build candidates from Stage 1 (raw) and Stage 2 (POS-mapped),
    convert each to a covered sentence range [lo,hi], merge overlaps,
    then dedupe across chunks by raw char-interval overlap.
    """
    spans_rel = sentence_spans(chunk)
    hits_added = 0

    candidates: List[dict] = []

    # ---- Stage 1: surface regexes on raw chunk ------------------------------
    for pname, pregex in STAGE1_REGEXES.items():
        for m in pregex.finditer(chunk):
            m_start_rel = m.start()
            m_end_rel   = m.end()
            rng_idx = _covered_sentence_range(spans_rel, m_start_rel, m_end_rel)
            if rng_idx is None:
                continue
            lo, hi = rng_idx
            candidates.append({
                "lo": lo,
                "hi": hi,
                "raw_start": abs_offset + m_start_rel,
                "raw_end":   abs_offset + m_end_rel,
                "rule": f"S1_{pname}",
            })

    # ---- Stage 2: POS regexes with offset mapping ---------------------------
    if STAGE2_REGEXES and HAS_POS_TAGGER:
        try:
            stream, pieces = tag_stream_with_offsets(chunk, 'verb')
            stream_starts = [p[0] for p in pieces]
            stream_ends   = [p[1] for p in pieces]

            def _stream_to_chunk_raw(ss: int, se: int) -> Tuple[int,int] | None:
                i = bisect_right(stream_ends, ss)
                j = bisect_left(stream_starts, se) - 1
                if i >= len(pieces) or j < i:
                    return None
                raw_s_rel = min(p[2] for p in pieces[i:j+1])
                raw_e_rel = max(p[3] for p in pieces[i:j+1])
                return raw_s_rel, raw_e_rel

            for pname, pregex in STAGE2_REGEXES.items():
                for m in pregex.finditer(stream):
                    mapres = _stream_to_chunk_raw(m.start(), m.end())
                    if not mapres:
                        continue
                    raw_s_rel, raw_e_rel = mapres
                    rng_idx = _covered_sentence_range(spans_rel, raw_s_rel, raw_e_rel)
                    if rng_idx is None:
                        continue
                    lo, hi = rng_idx
                    candidates.append({
                        "lo": lo,
                        "hi": hi,
                        "raw_start": abs_offset + raw_s_rel,
                        "raw_end":   abs_offset + raw_e_rel,
                        "rule": f"S2_{pname}",
                    })
        except Exception:
            pass

    # ---- Merge overlapping sentence ranges within this chunk ----------------
    blocks = _merge_sentence_blocks(candidates)

    # ---- Dedupe across chunks by raw char-interval overlap ------------------
    for blk in blocks:
        rs, re = blk["raw_start"], blk["raw_end"]
        if _ranges_overlap(covered_ranges, rs, re):
            continue  # already counted elsewhere
        _add_range(covered_ranges, rs, re)
        hits_added += 1

        # Sample representative text: the merged sentence block within this chunk
        cover_start = spans_rel[blk["lo"]][0]
        cover_end   = spans_rel[blk["hi"]][1]
        cover_text  = chunk[cover_start:cover_end].strip()
        _reservoir_maybe_add(
            heap, k, rng.random(),
            SampleHit(rule=blk["rule"], text=cover_text)
        )

    return hits_added


def process_file(path: Path, chunk_size: int, overlap: int, sample_size: int, seed: int) -> tuple[FileStats, List[Tuple[float, SampleHit]]]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}") from e

    text = normalize_text(raw)
    total_chars = len(text)
    if total_chars == 0:
        return FileStats(str(path), 0, 0, 0.0), []

    # independent RNG per file for reproducibility
    rng = random.Random((seed ^ hash(str(path))) & 0xFFFFFFFF)

    covered_ranges: List[Tuple[int,int]] = []
    total_hits = 0

    # local reservoir of size sample_size
    heap: List[Tuple[float, SampleHit]] = []

    step = max(1, chunk_size - overlap)
    for start in range(0, total_chars, step):
        end = min(total_chars, start + chunk_size)
        chunk = text[start:end]
        total_hits += _scan_chunk_for_hits(
            chunk=chunk,
            abs_offset=start,
            covered_ranges=covered_ranges,
            rng=rng,
            heap=heap,
            k=sample_size,
        )
        if end == total_chars:
            break

    rate = (total_hits * 1000.0) / total_chars if total_chars > 0 else 0.0

    # convert local max-heap of negatives into list of (key, item)
    samples: List[Tuple[float, SampleHit]] = [(-neg, item) for (neg, item) in heap]
    return FileStats(str(path), total_chars, total_hits, rate), samples

# ──────────────────────────────────────────────────────────────────────
# 3) Orchestration
# ──────────────────────────────────────────────────────────────────────

def find_txt_files(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*.txt")) if p.is_file()]

def run_parallel(paths: List[Path], workers: int, chunk_size: int, overlap: int, sample_size: int, sample_seed: int) -> Tuple[List[FileStats], List[Tuple[float, SampleHit]]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    per_file_stats: List[FileStats] = []
    # Global heap to merge samples from all files (store as negatives to reuse helper)
    gh: List[Tuple[float, SampleHit]] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(process_file, p, chunk_size, overlap, sample_size, sample_seed): p
            for p in paths
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Files"):
            p = futs[fut]
            try:
                stats, samples = fut.result()
            except Exception as e:
                print(f"[error] {p}: {e}", file=sys.stderr)
                continue
            per_file_stats.append(stats)
            # merge local samples into global reservoir using same policy
            for key, item in samples:
                _reservoir_maybe_add(gh, sample_size, key, item)

    # turn heap of negatives back into ascending keys
    merged_samples: List[Tuple[float, SampleHit]] = sorted([(-neg, item) for (neg, item) in gh], key=lambda x: x[0])
    return per_file_stats, merged_samples

def print_summary(per_file: List[FileStats], samples: List[Tuple[float, SampleHit]], sample_size: int) -> None:
    # Per-file lines
    for s in sorted(per_file, key=lambda x: x.rate_per_1000, reverse=True):
        print(f"{Path(s.file).name}\tchars={s.chars}\thits={s.hits}\trate_per_1k={s.rate_per_1000:.3f}")

    total_chars = sum(s.chars for s in per_file)
    total_hits = sum(s.hits for s in per_file)
    overall = (total_hits * 1000.0) / total_chars if total_chars > 0 else 0.0

    print("\nHUMAN_BASELINE")
    print(f"rate_per_1k={overall:.3f}\thits={total_hits}\tchars={total_chars}")

    # Samples
    n = min(sample_size, len(samples))
    if n > 0:
        print(f"\nSAMPLES n={n}")
        # shuffle-free deterministic order by key; print rule and text only
        for i, (_, hit) in enumerate(samples[:n], 1):
            frag = " ".join(hit.text.split())
            print(f"{i:03d}. [{hit.rule}] {frag}")

# ──────────────────────────────────────────────────────────────────────
# 4) CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute human baseline from .txt files.")
    ap.add_argument("--dir", type=str, default="human_writing_samples",
                    help="Root directory containing .txt files")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                    help="Number of worker processes")
    ap.add_argument("--chunk-size", type=int, default=20_000,
                    help="Chunk size in characters for regex ingestion")
    ap.add_argument("--overlap", type=int, default=500,
                    help="Overlap in characters between chunks for boundary safety")
    ap.add_argument("--sample-size", type=int, default=300,
                    help="Number of random hits to print across all files")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="Seed for sampling reproducibility")
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    root = Path(args.dir)
    if not root.exists():
        print(f"Directory not found: {root}", file=sys.stderr)
        sys.exit(2)

    paths = find_txt_files(root)
    if not paths:
        print(f"No .txt files under {root}", file=sys.stderr)
        sys.exit(1)

    _ = list(tqdm(paths, desc="Enumerating files", total=len(paths)))
    stats, samples = run_parallel(
        paths,
        workers=args.workers,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if not stats:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    print_summary(stats, samples, args.sample_size)

if __name__ == "__main__":
    main()
