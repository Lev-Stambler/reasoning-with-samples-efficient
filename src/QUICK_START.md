# Beam Search Quick Start (Python 3.12 + uv)

## Setup

This project uses **uv** package manager with **Python 3.12**.

```bash
cd /Users/nisargdesai/Documents/xai-hackathon/reasoning-with-samples-efficient

# Set your Grok API key
export XAI_API_KEY="your-key-here"
# Or add to .env file
echo "XAI_API_KEY=your-key-here" > .env
```

## Test Beam Search

```bash
cd src

# Test basic functionality (uv automatically manages dependencies)
uv run --python 3.12 python test_beam_search.py
```

## Run Benchmark

### Enable Beam Search Only
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    mcmc.enabled=false \
    greedy.enabled=false
```

### Compare Beam Search with MCMC
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    mcmc.enabled=true
```

### Customize Parameters
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.alpha=4.0 \
    beam_search.beam_width=5 \
    beam_search.tokens_per_step=192 \
    beam_search.debug=true \
    benchmark.num_problems=10
```

## Configuration

Edit `src/conf/config.yaml` or override via command line:

```yaml
beam_search:
  enabled: false           # Set to true to use
  alpha: 4.0              # Power factor (π(x) = p(x)^α)
  beam_width: 5           # Keep top-5 beams after pruning
  n_per_beam: 5           # Generate 5 continuations per beam (TRUE beam search!)
  tokens_per_step: 192    # Tokens per expansion
  length_penalty: 0.6     # Length normalization
  proposal_temperature: 1.0
  top_logprobs: 5
  debug: false            # Verbose output
```

**Note**: With `beam_width=5` and `n_per_beam=5`, each iteration generates **25 candidates** (5 beams × 5 samples), then prunes to the **top 5 beams**. This is TRUE beam search with proper branching!

## Quick Parameter Guide

### Fast/Cheap
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=3 \
    beam_search.tokens_per_step=256 \
    benchmark.num_problems=5
```

### Best Quality
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=10 \
    beam_search.tokens_per_step=50 \
    beam_search.alpha=6.0
```

### Debug Mode
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.debug=true \
    benchmark.num_problems=2
```

## Key Parameters

- **`alpha`**: Power scaling factor (1.0-10.0, default 4.0)
  - Higher = prefer high-probability sequences more
- **`beam_width`**: Number of beams to keep after pruning (1-20, default 5)
  - Final number of parallel hypotheses
- **`n_per_beam`**: Generate n samples per beam (1-10, default 5)
  - Controls branching factor: beam_width × n_per_beam = total candidates
  - Higher = better exploration but more expensive
- **`tokens_per_step`**: Chunk size (50-512, default 192)
  - Smaller = fine-grained, more expansions
- **`length_penalty`**: Normalization (0.0-1.0, default 0.6)
  - Higher = prefer longer sequences

### Understanding n_per_beam:
- `n_per_beam=1`: Pseudo-beam search (each beam → 1 continuation)
- `n_per_beam=5`: TRUE beam search (each beam → 5 continuations)
- With `beam_width=5, n_per_beam=5`: **25 candidates per iteration**, prune to top 5

## Expected Output

```
================================================================================
HUMANEVAL BENCHMARK RESULTS
================================================================================
┌────────────┬──────────────┬─────────────────────────┬──────────────┐
│ Benchmark  │ Model        │ Strategy                │ Pass Rate (%)│
├────────────┼──────────────┼─────────────────────────┼──────────────┤
│ HumanEval  │ grok-2-1212  │ BeamSearch(α=4,w=5,...) │ 78.0%        │
│ HumanEval  │ grok-2-1212  │ MCMC(α=4,steps=10,...)  │ 75.0%        │
│ HumanEval  │ grok-2-1212  │ Greedy                  │ 70.0%        │
└────────────┴──────────────┴─────────────────────────┴──────────────┘
```

## Troubleshooting

### API Key Error
```bash
echo $XAI_API_KEY  # Check it's set
```

### Rate Limiting
```bash
# Reduce API calls
uv run --python 3.12 python run_benchmark.py \
    beam_search.beam_width=3 \
    beam_search.tokens_per_step=256
```

### Debugging
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.debug=true \
    benchmark.num_problems=1
```

## More Info

- Full documentation: `src/BEAM_SEARCH_API_README.md`
- Implementation: `src/benchmark_runner.py` (line 321)
- Config: `src/conf/config.yaml`
