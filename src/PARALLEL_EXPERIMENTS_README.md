# Running Parallel Experiments

## Quick Start

The `beam_search_evals.sh` script automatically runs multiple beam search experiments in parallel.

```bash
cd src
./beam_search_evals.sh
```

## Features

✅ **Auto-detects OS** - Works on macOS and Linux  
✅ **Auto-installs GNU parallel** - Checks if already installed  
✅ **Runs 5 experiments in parallel** - Tests different configurations  
✅ **Compares with greedy baseline** - Side-by-side comparison  

## What It Does

The script runs these 5 experiments simultaneously:

| Experiment | beam_width | n_per_beam | alpha | Description |
|------------|-----------|------------|-------|-------------|
| 1 | 2 | 2 | 4.0 | Baseline (small beam) |
| 2 | 3 | 2 | 4.0 | Larger beam width |
| 3 | 2 | 3 | 4.0 | More continuations per beam |
| 4 | 5 | 2 | 4.0 | Very large beam |
| 5 | 2 | 2 | 6.0 | Higher alpha (more confident) |

All experiments also run **greedy sampling** for comparison.

## Output

### Experiment Logs
Each experiment saves its full output to `src/eval_logs/`:
- `exp_beam_width=2_n_per_beam=2_alpha=4.0.log`
- `exp_beam_width=3_n_per_beam=2_alpha=4.0.log`
- etc.

### Prediction Files
Each experiment also saves predictions to `src/predictions/`:
- `gsm8k_grok_4_1_fast_non_reasoning_BeamSearchalpha4.0_beam_width2_n_per_beam2.jsonl`
- `gsm8k_grok_4_1_fast_non_reasoning_Greedy.jsonl`
- etc.

### Aggregated Results
At the end, the script automatically aggregates all results:

```
==================================================================================================================
AGGREGATED BENCHMARK RESULTS FROM ALL EXPERIMENTS
==================================================================================================================

+---------------------------+------------------------------------+-----------+--------------+--------------+------------+-------------+--------------+----------+
| Experiment                | Strategy                           | Pass Rate | Avg Time (s) | Total Tokens | Avg Tokens | Total Cost  | Cost/Problem | Problems |
+===========================+====================================+===========+==============+==============+============+=============+==============+==========+
| beam_width=2_n_per_beam=2 | BeamSearch(α=4.0,width=2,n=2)     | 60.0%     | 1.67         | 8,371        | 837.1      | $0.0025     | $0.0003      | 10       |
+---------------------------+------------------------------------+-----------+--------------+--------------+------------+-------------+--------------+----------+
| beam_width=2_n_per_beam=2 | Greedy                             | 40.0%     | 1.41         | 3,813        | 381.3      | $0.0011     | $0.0001      | 10       |
+---------------------------+------------------------------------+-----------+--------------+--------------+------------+-------------+--------------+----------+
...
```

**Summary automatically shows:**
- 🏆 Best performing beam search configuration
- 📊 Average greedy baseline
- 📈 Improvement over greedy

## Customization

Edit `params.txt` in the script to run different configurations:

```bash
# Edit the script to add/modify experiments
cat > params.txt << 'EOF'
beam_width=2 n_per_beam=2 alpha=4.0
beam_width=10 n_per_beam=5 alpha=8.0  # Add your own!
EOF
```

Or directly modify the script:
```bash
# Change benchmark settings
benchmark.name=gsm8k          # or humaneval, swebench
benchmark.num_problems=10     # number of problems
benchmark.max_tokens=128      # max tokens per completion

# Change parallelization
parallel -j 4    # max 4 jobs at once
parallel -j 8    # max 8 jobs at once
```

## Requirements

- **GNU parallel** - Auto-installed by script
- **uv** - Python package manager
- **XAI_API_KEY** - Set in environment

## Supported OS

- ✅ macOS (via Homebrew)
- ✅ Linux (apt-get, yum, dnf)
- ✅ Windows (WSL)

## Manual Installation

If auto-install fails:

**macOS:**
```bash
brew install parallel
```

**Ubuntu/Debian:**
```bash
sudo apt-get install parallel
```

**RHEL/CentOS/Fedora:**
```bash
sudo yum install parallel
# or
sudo dnf install parallel
```

## Example Run

```bash
cd src
./beam_search_evals.sh
```

Output:
```
Checking for GNU parallel...
✓ GNU parallel is already installed
GNU parallel 20251122

Running 5 experiments in parallel (max 4 at once)...

============================================================
CONFIGURATION
============================================================
...

Problem 1/10: HumanEval/0
  Testing BeamSearch(alpha=4.0, beam_width=2, n_per_beam=2)... ✓
  Testing Greedy... ✓
...

============================================================
BENCHMARK RESULTS
============================================================
Strategy                    | Pass Rate | Avg Time | Total Tokens | Total Cost
---------------------------|-----------|----------|--------------|------------
BeamSearch(alpha=4.0)      |     0.00% |   12.34s |       45,678 |     $0.123
Greedy                     |     0.00% |    5.67s |       23,456 |     $0.067
```

## Manual Result Aggregation

To re-aggregate results after experiments complete:

```bash
cd src
python aggregate_results.py eval_logs
```

Or with uv:
```bash
uv run --python 3.12 python aggregate_results.py eval_logs
```

This will:
- Parse all log files in `eval_logs/`
- Extract benchmark results
- Create unified comparison table
- Show best configurations
- Calculate improvements

## Troubleshooting

**"parallel: command not found"**
- Script should auto-install, but if it fails, install manually (see above)

**"Homebrew not found"** (macOS)
- Install Homebrew first:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**"No supported package manager found"** (Linux)
- Install manually using your package manager
- Or download from: https://www.gnu.org/software/parallel/

**Rate limiting**
- Reduce parallel jobs: `parallel -j 2` instead of `-j 4`
- Add delays between problems in the script

## Advanced Usage

### Run Different Benchmarks

```bash
# Edit the script and change:
benchmark.name=humaneval          # instead of gsm8k
benchmark.num_problems=20         # more problems
benchmark.max_tokens=512          # more tokens for math
```

### Run More Experiments

```bash
# Extend params.txt with more configurations:
cat >> params.txt << 'EOF'
beam_width=7 n_per_beam=3 alpha=5.0
beam_width=10 n_per_beam=10 alpha=10.0
EOF
```

### Sequential Execution

If you don't want parallel execution:
```bash
parallel -j 1 ...   # Run one at a time
```

### Save Logs

```bash
./beam_search_evals.sh 2>&1 | tee experiments.log
```

## Performance Tips

1. **Start small** - Test with 2-5 problems first
2. **Monitor API usage** - Parallel runs use more quota
3. **Adjust parallelization** - `-j 2` for slower connections
4. **Use async beam search** - Already enabled for 2-10× speedup per experiment

## See Also

- `ASYNC_BEAM_SEARCH_README.md` - Details on async parallelization
- `TRUE_BEAM_SEARCH_IMPLEMENTATION.md` - Beam search algorithm
- `src/conf/config.yaml` - All configuration options
