# Not-X-But-Y Benchmark

A benchmark measuring a specific kind of slop that LLMs have been converging on: "Not X, but Y" or "It's not just X, it's Y" and similar constructions. These are not bad writing in themselves, but become perceived as repetitive or distinctly AI-generated when they appear far more frequently than in human writing. Some examples taken from AI writing:

## Quoted examples

> "The silence after Elan snatched the notebook wasn't empty. It was thick."

> "The Sole of Destiny isn't a weapon; it's a responsibility."

> "This is not a local anomaly. It's global."

> "It's not just performance art. It's… manipulation."

> "Shoes don't remember who ran first. They remember who limped last."

> "It is not language as I know it; it is a pattern of resonance that my brain is forced to interpret."

> "For the first time in years, the silence doesn't feel empty. It feels shared."

> "I'll teach kids that power isn't about the shoes you wear, but the choices you make."

> "The translation is not merely semantic. It's ethical architecture."

This evaluated model is prompted to write 1000 works to 300 different writing prompts. The text is then passed through a number of Regex expressions to capture the different ways "Not-x-but-y" phrases manifest. The final score is the frequency of these phrases in the text, per 1,000 characters.

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

MIT License
