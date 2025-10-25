# Not-X-But-Y Benchmark

A benchmark for measuring "AI slop" patterns in language model outputs, specifically the overuse of phrases like "not X, but Y" that are characteristic of artificial writing.

## Quickstart

```bash
# 1. Install dependencies
uv venv
. .venv/bin/activate
uv pip install -e .
python -m spacy download en_core_web_sm

# 2. Configure API credentials
cp env.example .env
# Edit .env with your API key and endpoint

# 3. Run the benchmark
python main.py claude-sonnet-4-20250514 --results results/claude-sonnet-4.json --n-prompts 300

# 4. Generate leaderboard
python create_results_chart.py
# Creates leaderboard.png with all model results
```

---

## What This Benchmark Measures

This benchmark detects overuse of rhetorical contrast patterns that are common in AI-generated text but rare in human writing. These patterns include:

- **"Not X, but Y"** constructions: "It's not a bug, but a feature"
- **Contrasts with dashes**: "He wasn't angry—he was furious"
- **Gerund fragments**: "Not running, walking carefully"
- **35+ additional patterns** identified through linguistic analysis

The benchmark computes a **slop rate**: hits per 1,000 characters. Lower is better.

### Human Baseline

The human baseline of **0.065** was computed from 82 published books (~53M characters) including:
- Fiction (literary, sci-fi, mystery, fantasy)
- Non-fiction (memoirs, history, essays)

This represents the natural rate of "Not X but Y" constructions in human-authored novels.

---

## How the Evaluation Works

### 1. Generation Phase

The benchmark:
1. Sends 300 creative writing prompts to your model
2. Asks for ~1,000 word responses
3. Uses temperature=0.7, min_p=0.1 (when supported)
4. Runs with configurable parallelism (default: 8 workers)

```bash
python main.py <model-name> \
  --results results/<model>.json \
  --n-prompts 300 \
  --workers 8 \
  --max-retries 5
```

### 2. Scoring Phase

Each output is analyzed using a **two-stage pattern matching system**:

**Stage 1: Surface Patterns (10 regexes)**
- Detects obvious patterns like "not X, but Y"
- Fast surface-level matching
- Catches ~40% of instances

**Stage 2: POS-Tagged Patterns (35 regexes)**
- Uses spaCy part-of-speech tagging
- Detects subtle grammatical patterns
- Handles variations like "doesn't X, Y's" or "not just X; Y"
- Catches the remaining ~60%

**Deduplication:**
- Matches are merged at sentence boundaries
- Overlapping sentence ranges count as one hit
- Cross-sentence patterns are supported

### 3. Results Format

Results are saved progressively to JSON:

```json
{
  "model-name": {
    "test_model": "model-name",
    "endpoint": "https://api.example.com/v1/messages",
    "params": { "temperature": 0.7, ... },
    "samples": [
      {
        "prompt_index": 0,
        "prompt": "Write about...",
        "output": "Generated text...",
        "chars": 5234,
        "hits": 3,
        "rate_per_1k": 0.573
      }
    ],
    "summary": {
      "total_prompts": 300,
      "total_chars": 1567890,
      "total_hits": 987,
      "rate_per_1k": 0.629
    }
  }
}
```

---

## API Configuration

The benchmark auto-detects API types based on your `BASE_URL`:

### Anthropic
```bash
BASE_URL="https://api.anthropic.com/v1/messages"
API_KEY="sk-ant-..."
```
```bash
python main.py claude-sonnet-4-20250514 --results results/claude.json
```

### OpenRouter
```bash
BASE_URL="https://openrouter.ai/api/v1/chat/completions"
API_KEY="sk-or-..."
```
```bash
python main.py google/gemma-3-4b-it --results results/gemma.json
```

### OpenAI-Compatible
```bash
BASE_URL="https://api.openai.com/v1/chat/completions"
API_KEY="sk-..."
```
```bash
python main.py gpt-4o --results results/gpt4o.json
```

---

## Additional Tools

### Compute Human Baseline

Recalculate the human baseline from the included text samples:

```bash
# Basic
python human_baseline.py

# With per-file breakdown
python human_baseline.py -v
```

### Generate Charts

Create a leaderboard visualization:

```bash
python create_results_chart.py
# Output: leaderboard.png
```

Customize the output:
```bash
python create_results_chart.py \
  --results-dir my-results/ \
  --output my-leaderboard.png
```

---

## Project Structure

```
.
├── main.py                    # Main evaluation runner
├── src/
│   ├── api.py                # API client (OpenAI/Anthropic/OpenRouter)
│   ├── scorer.py             # Pattern matching and scoring
│   ├── regexes_v3.py         # Stage 1 surface patterns (10)
│   ├── regexes_pos.py        # Stage 2 POS patterns (35)
│   ├── pos_tagger.py         # spaCy integration
│   └── io_utils.py           # Thread-safe JSON updates
├── human_baseline.py         # Compute baseline from books
├── create_results_chart.py   # Generate leaderboard
├── recalc.py                 # Recalculate existing scores
├── prompts.json              # 300 creative writing prompts
├── human_writing_samples/    # 82 books for baseline (~53MB)
└── results/                  # Evaluation results (JSON)
```

---

## Scoring Validation

The scorer has strict validation to prevent silent failures:

- **45 patterns required**: 10 Stage 1 + 35 Stage 2
- **Module-level validation**: Runs on import
- **No silent fallbacks**: Missing regexes cause immediate errors
- **Pattern verification**: Ensures all are compiled regex objects

If imports fail, you'll see:
```
ImportError: Failed to load Stage1 regexes from regexes_v3: ...
This is a critical error - scoring will be incorrect without proper regexes.
```

---

## Features

- **Parallel generation** with configurable workers
- **Automatic retry logic** with exponential backoff
- **Progressive results** - saved after each completion
- **Thread-safe writes** - safe to run multiple instances
- **Resume support** - rerun with same args to continue
- **Multiple API types** - OpenAI, Anthropic, OpenRouter
- **Cross-sentence matching** - handles multi-sentence patterns
- **Sentence-level deduplication** - prevents double-counting

---

## CLI Reference

### Main Evaluation
```bash
python main.py <model-name> [options]

Required:
  model                    Model identifier (e.g., gpt-4o, claude-sonnet-4-20250514)

Options:
  --prompts FILE          Prompts JSON file (default: prompts.json)
  --results FILE          Results JSON file (default: results.json)
  --workers N             Parallel workers (default: 8)
  --timeout SECS          HTTP timeout (default: 480.0)
  --max-tokens N          Max tokens per generation (default: 8096)
  --n-prompts N           Number of prompts to use (default: 300)
  --max-retries N         Max retries per request (default: 5)
  --retry-delay SECS      Delay between retries (default: 5.0)
```

### Human Baseline
```bash
python human_baseline.py [options]

Options:
  --samples-dir DIR       Directory with .txt files (default: human_writing_samples/)
  -v, --verbose          Show per-file statistics
```

### Chart Generation
```bash
python create_results_chart.py [options]

Options:
  --results-dir DIR       Directory with results JSON (default: results/)
  --output FILE          Output image path (default: leaderboard.png)
```

---

## Environment Variables

Create a `.env` file or set environment variables:

```bash
BASE_URL=https://api.anthropic.com/v1/messages
API_KEY=sk-ant-api03-...
```

Both `BASE_URL` and `API_KEY` are required.

---

## Example Workflow

```bash
# 1. Test a model quickly (10 prompts for debugging)
python main.py claude-sonnet-4-20250514 \
  --results results/claude-test.json \
  --n-prompts 10

# 2. Full evaluation (300 prompts)
python main.py claude-sonnet-4-20250514 \
  --results results/claude-full.json \
  --n-prompts 300 \
  --workers 16

# 3. Evaluate multiple models
for model in "gpt-4o" "claude-sonnet-4-20250514" "gemini-2.5-flash"; do
  python main.py $model --results results/${model}.json
done

# 4. Generate leaderboard
python create_results_chart.py
open leaderboard.png
```

---

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{notxbuty2025,
  title={Not-X-But-Y-Bench},
  author={Sam Paech},
  year={2025},
  url={https://github.com/sam-paech/not-x-but-y-bench}
}
```

---

## License

MIT License - see LICENSE file for details
