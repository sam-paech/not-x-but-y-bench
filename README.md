
## Overview
This tool generates ~1,000-word completions for a list of prompts across an OpenAI-compatible endpoint, then scores each output for “not-x-but-y” contrast usage with cross-sentence aware deduping. It saves results incrementally and atomically into a single JSON file keyed by model id.

## Features
- Parallel generation with threads and retry logic.
- Auto-detects API flavor by `BASE_URL`:
  - OpenRouter: OpenAI-compatible, supports `min_p`.
  - OpenAI-compatible: standard `/v1/chat/completions`.
  - Anthropic: Messages API at `/v1/messages`.
- Fixed generation params: `temperature=0.7`, `min_p=0.1` when supported.
- Cross-sentence matching and dedupe for the “not-x-but-y” score.
- Progressive, thread-safe, atomic results writes with resume support.
- TQDM progress.

## Install with uv
```bash
# Create an isolated environment
uv venv
. .venv/bin/activate

# Install project and deps
uv pip install -e .

# If you will use Stage-2 POS rules, install the spaCy model:
python -m spacy download en_core_web_sm
