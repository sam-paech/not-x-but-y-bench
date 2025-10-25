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
    from . import regexes_v3
    STAGE1_REGEXES: Dict[str, re.Pattern] = regexes_v3.compiled
    if len(STAGE1_REGEXES) != 10:
        raise ValueError(f"Expected 10 Stage1 regexes, got {len(STAGE1_REGEXES)}")
except Exception as e:
    raise ImportError(f"Failed to load Stage1 regexes from regexes_v3: {e}. "
                      "This is a critical error - scoring will be incorrect without proper regexes.")

# --- Stage-2 regexes (POS) ---
try:
    from . import regexes_pos
    STAGE2_REGEXES: Dict[str, re.Pattern] = {
        "POS_DOESNT_VERB": regexes_pos.RE_POS_DOESNT_VERB,
        "POS_DONT_JUST_VERB": regexes_pos.RE_POS_DONT_JUST_VERB,
        "POS_GERUND_FRAGMENT": regexes_pos.RE_POS_GERUND_FRAGMENT,
        "POS_NOT_ADJ": regexes_pos.RE_POS_NOT_ADJ,
        "POS_DASH_VERB": regexes_pos.RE_POS_DASH_VERB,
        "POS_NOT_JUST_VERB_PAST": regexes_pos.RE_POS_NOT_JUST_VERB_PAST,
        "POS_COLON_VERB": regexes_pos.RE_POS_COLON_VERB,
        "POS_ISNT_JUST_VERB": regexes_pos.RE_POS_ISNT_JUST_VERB,
        "POS_QUOTE_MULTI_VERB": regexes_pos.RE_POS_QUOTE_MULTI_VERB,
        "POS_ELLIPSIS_VERB": regexes_pos.RE_POS_ELLIPSIS_VERB,
        "POS_NOT_NOUN": regexes_pos.RE_POS_NOT_NOUN,
        "POS_DOESNT_VERB_EMPHASIS": regexes_pos.RE_POS_DOESNT_VERB_EMPHASIS,
        "POS_DASH_VERB_BROAD": regexes_pos.RE_POS_DASH_VERB_BROAD,
        "POS_ELLIPSIS_BROAD": regexes_pos.RE_POS_ELLIPSIS_BROAD,
        "POS_NOT_BECAUSE": regexes_pos.RE_POS_NOT_BECAUSE,
        "POS_GERUND_BROAD": regexes_pos.RE_POS_GERUND_BROAD,
        "POS_QUOTE_VERBING": regexes_pos.RE_POS_QUOTE_VERBING,
        "POS_DOESNT_LITERAL": regexes_pos.RE_POS_DOESNT_LITERAL,
        "POS_DASH_NOUN_SWAP": regexes_pos.RE_POS_DASH_NOUN_SWAP,
        "POS_ISNT_DASH_EMPHASIS": regexes_pos.RE_POS_ISNT_DASH_EMPHASIS,
        "POS_THATS_NOT_NOUN": regexes_pos.RE_POS_THATS_NOT_NOUN,
        "POS_GERUND_EMPHASIS": regexes_pos.RE_POS_GERUND_EMPHASIS,
        "POS_QUOTE_ATTRIBUTION_VERB": regexes_pos.RE_POS_QUOTE_ATTRIBUTION_VERB,
        "POS_ISNT_NOUN": regexes_pos.RE_POS_ISNT_NOUN,
        "POS_ITS_NOT_JUST": regexes_pos.RE_POS_ITS_NOT_JUST,
        "POS_DASH_GERUND_OBJ": regexes_pos.RE_POS_DASH_GERUND_OBJ,
        "POS_ELLIPSIS_DIALOGUE": regexes_pos.RE_POS_ELLIPSIS_DIALOGUE,
        "POS_SEMI_NOUN": regexes_pos.RE_POS_SEMI_NOUN,
        "POS_ISNT_ADJ_NOUN": regexes_pos.RE_POS_ISNT_ADJ_NOUN,
        "POS_DIALOGUE_ATTR": regexes_pos.RE_POS_DIALOGUE_ATTR,
        "POS_TO_VERB_ISNT": regexes_pos.RE_POS_TO_VERB_ISNT,
        "POS_I_AM_NOT_SEMI": regexes_pos.RE_POS_I_AM_NOT_SEMI,
        "POS_NOT_ANYMORE_ITS": regexes_pos.RE_POS_NOT_ANYMORE_ITS,
        "POS_AINT_SIMPLE": regexes_pos.RE_POS_AINT_SIMPLE,
    }
    # Stage 2b: Lemma-based regexes
    STAGE2_REGEXES["LEMMA_SAME_VERB"] = regexes_pos.RE_LEMMA_SAME_VERB
    if len(STAGE2_REGEXES) != 35:
        raise ValueError(f"Expected 35 Stage2 regexes, got {len(STAGE2_REGEXES)}")
except Exception as e:
    raise ImportError(f"Failed to load Stage2 regexes from regexes_pos: {e}. "
                      "This is a critical error - scoring will be incorrect without proper regexes.")

# POS mapping with raw offsets
try:
    from .pos_tagger import tag_stream_with_offsets
    HAS_POS = True
except Exception as e:
    raise ImportError(f"Failed to load pos_tagger: {e}. "
                      "This is a critical error - Stage2 POS regexes won't work without the tagger.")

from bisect import bisect_left, bisect_right

# Validation: Ensure all expected regexes are loaded
def _validate_regexes():
    """Validate that all expected regex patterns are loaded correctly."""
    expected_stage1 = 10
    expected_stage2 = 35

    if len(STAGE1_REGEXES) != expected_stage1:
        raise RuntimeError(
            f"CRITICAL: Stage1 regex count mismatch. Expected {expected_stage1}, got {len(STAGE1_REGEXES)}. "
            f"This will cause incorrect scoring results."
        )

    if len(STAGE2_REGEXES) != expected_stage2:
        raise RuntimeError(
            f"CRITICAL: Stage2 regex count mismatch. Expected {expected_stage2}, got {len(STAGE2_REGEXES)}. "
            f"This will cause incorrect scoring results."
        )

    # Verify all patterns are actually compiled regex objects
    for name, pattern in STAGE1_REGEXES.items():
        if not hasattr(pattern, 'finditer'):
            raise RuntimeError(f"CRITICAL: Stage1 pattern '{name}' is not a compiled regex")

    for name, pattern in STAGE2_REGEXES.items():
        if not hasattr(pattern, 'finditer'):
            raise RuntimeError(f"CRITICAL: Stage2 pattern '{name}' is not a compiled regex")

# Run validation on module import
_validate_regexes()


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
    if STAGE2_REGEXES:
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
