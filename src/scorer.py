# src/scorer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# --- Normalization ---
def normalize_text(text: str) -> str:
    for old, new in {'“':'"', '”':'"', '‘':"'", '’':"'", '—':'-', '–':'-'}.items():
        text = text.replace(old, new)
    return text

# --- Sentence spans ---
_SENT_SPLIT = re.compile(r'[^.!?]*[.!?]', flags=re.S)
def sentence_spans(text: str) -> List[Tuple[int, int]]:
    spans, last_end = [], 0
    for m in _SENT_SPLIT.finditer(text):
        spans.append((m.start(), m.end()))
        last_end = m.end()
    if last_end < len(text):
        spans.append((last_end, len(text)))
    return spans

# --- Stage-1 regexes (surface) ---
try:
    import regexes_v3
    STAGE1_REGEXES: Dict[str, re.Pattern] = regexes_v3.compiled
except Exception:
    # Fallback simplified patterns
    STAGE1_REGEXES = {
        "RE_NOT_BUT": re.compile(r"\bnot\b(?:(?![.?!]).){1,160}?\bbut\b", re.I | re.S),
        "RE_NT_BUT": re.compile(r"\b\w+n't\b(?:(?![.?!]).){1,160}?\bbut\b", re.I | re.S),
    }

# --- Stage-2 regexes (POS) ---
try:
    import regexes_pos
    STAGE2_REGEXES: Dict[str, re.Pattern] = {
        "POS_DOESNT_VERB": regexes_pos.RE_POS_DOESNT_VERB,
        "POS_GERUND_FRAGMENT": regexes_pos.RE_POS_GERUND_FRAGMENT,
        "POS_DIALOGUE_ATTR": regexes_pos.RE_POS_DIALOGUE_ATTR,
        "POS_I_AM_NOT_SEMI": regexes_pos.RE_POS_I_AM_NOT_SEMI,
        "POS_NOT_ANYMORE_ITS": regexes_pos.RE_POS_NOT_ANYMORE_ITS,
    }
except Exception:
    STAGE2_REGEXES = {}

# POS mapping with raw offsets
try:
    from pos_tagger import tag_stream_with_offsets
    HAS_POS = True
except Exception:
    HAS_POS = False
    def tag_stream_with_offsets(text: str, pos_type: str = 'verb'):
        return text, [(0, len(text), 0, len(text))]

from bisect import bisect_left, bisect_right


def _covered_sentence_range(spans: List[Tuple[int,int]], start: int, end: int):
    if not spans or start >= end:
        return None
    starts = [s for s, _ in spans]
    ends   = [e for _, e in spans]
    lo = bisect_right(ends, start)
    hi = bisect_left(starts, end) - 1
    if lo >= len(spans) or hi < 0 or lo > hi:
        return None
    return lo, hi


def _merge_intervals(items: List[dict]) -> List[dict]:
    if not items:
        return []
    items_sorted = sorted(items, key=lambda d: (d['lo'], d['hi'], d['raw_start']))
    merged = []
    cur = items_sorted[0].copy()
    for it in items_sorted[1:]:
        if it['lo'] <= cur['hi']:
            cur['hi'] = max(cur['hi'], it['hi'])
            cur['raw_end'] = max(cur['raw_end'], it['raw_end'])
        else:
            merged.append(cur)
            cur = it.copy()
    merged.append(cur)
    return merged


def extract_contrast_matches_unique(text: str) -> List[Dict[str, str]]:
    t_norm = normalize_text(text)
    spans = sentence_spans(t_norm)

    candidates: List[dict] = []

    # Stage 1 on raw
    for pname, pregex in STAGE1_REGEXES.items():
        for m in pregex.finditer(t_norm):
            rs, re_ = m.start(), m.end()
            rng = _covered_sentence_range(spans, rs, re_)
            if rng is None:
                continue
            lo, hi = rng
            candidates.append({
                "lo": lo, "hi": hi,
                "raw_start": rs, "raw_end": re_,
                "pattern_name": f"S1_{pname}",
                "match_text": m.group(0).strip(),
            })

    # Stage 2 on POS-tagged stream (map back to raw)
    if STAGE2_REGEXES and HAS_POS:
        try:
            stream, pieces = tag_stream_with_offsets(t_norm, 'verb')
            stream_starts = [p[0] for p in pieces]
            stream_ends   = [p[1] for p in pieces]

            def _stream_to_raw(ss: int, se: int):
                i = bisect_right(stream_ends, ss)
                j = bisect_left(stream_starts, se) - 1
                if i >= len(pieces) or j < i:
                    return None
                raw_s = min(p[2] for p in pieces[i:j+1])
                raw_e = max(p[3] for p in pieces[i:j+1])
                return raw_s, raw_e

            for pname, pregex in STAGE2_REGEXES.items():
                for m in pregex.finditer(stream):
                    mapres = _stream_to_raw(m.start(), m.end())
                    if not mapres:
                        continue
                    rs, re_ = mapres
                    rng = _covered_sentence_range(spans, rs, re_)
                    if rng is None:
                        continue
                    lo, hi = rng
                    candidates.append({
                        "lo": lo, "hi": hi,
                        "raw_start": rs, "raw_end": re_,
                        "pattern_name": f"S2_{pname}",
                        "match_text": t_norm[rs:re_].strip(),
                    })
        except Exception:
            pass

    merged = _merge_intervals(candidates)

    results: List[Dict[str, str]] = []
    for it in merged:
        s_lo, s_hi = it["lo"], it["hi"]
        block_start = spans[s_lo][0]
        block_end   = spans[s_hi][1]
        results.append({
            "sentence": t_norm[block_start:block_end].strip(),
            "pattern_name": it["pattern_name"],
            "match_text": it["match_text"],
        })
    return results


def score_text(text: str) -> Dict[str, float]:
    t = normalize_text(text)
    hits = extract_contrast_matches_unique(t)
    chars = len(t)
    rate = (len(hits) * 1000.0 / chars) if chars > 0 else 0.0
    return {"hits": len(hits), "chars": chars, "rate_per_1k": rate}
