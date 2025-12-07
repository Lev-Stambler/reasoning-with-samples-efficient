# Beam Search Eval Script Updates

## What Was Changed

Updated `beam_search_evals.sh` to automatically aggregate and display results from all parallel experiments in a unified format.

## New Features

### 1. **Automatic Log Management**
- Creates `eval_logs/` directory for all experiment outputs
- Each experiment saves to separate log file: `exp_beam_width=X_n_per_beam=Y_alpha=Z.log`
- Easy to track which experiment is which

### 2. **Python-Based Result Aggregation**
- New script: `aggregate_results.py`
- Parses all log files automatically
- Extracts benchmark result tables
- Creates unified comparison view

### 3. **Smart Result Display**

**Unified Table:**
```
==================================================================================================================
AGGREGATED BENCHMARK RESULTS FROM ALL EXPERIMENTS
==================================================================================================================

+---------------------------+------------------------------------+-----------+--------------+--------------+
| Experiment                | Strategy                           | Pass Rate | Avg Time (s) | Total Tokens |
+===========================+====================================+===========+==============+==============+
| beam_width=2_n_per_beam=2 | BeamSearch(α=4.0,width=2,n=2)     | 60.0%     | 1.67         | 8,371        |
+---------------------------+------------------------------------+-----------+--------------+--------------+
| beam_width=3_n_per_beam=2 | BeamSearch(α=4.0,width=3,n=2)     | 65.0%     | 2.34         | 12,456       |
+---------------------------+------------------------------------+-----------+--------------+--------------+
...
```

**Automatic Summary:**
```
==================================================================================================================
SUMMARY
==================================================================================================================

🏆 Best Beam Search Configuration:
   Strategy:   BeamSearch(α=4.0,width=3,n=2)
   Pass Rate:  65.0%
   Avg Time:   2.34s
   Total Cost: $0.0037

📊 Greedy Baseline (averaged across runs):
   Pass Rate:  42.0%
   Avg Time:   1.41s

📈 Improvement over Greedy: +23.0%
```

### 4. **Fallback Mode**
If Python aggregator isn't available, script falls back to simple AWK-based extraction.

## File Structure

```
src/
├── beam_search_evals.sh           # Main script
├── aggregate_results.py           # Result aggregator
├── eval_logs/                     # Experiment logs (created by script)
│   ├── exp_beam_width=2_n_per_beam=2_alpha=4.0.log
│   ├── exp_beam_width=3_n_per_beam=2_alpha=4.0.log
│   └── ...
└── predictions/                   # Prediction files (created by benchmark)
    ├── gsm8k_..._BeamSearch...jsonl
    └── gsm8k_..._Greedy.jsonl
```

## Usage

### Run All Experiments
```bash
cd src
./beam_search_evals.sh
```

### Re-Aggregate Results
```bash
cd src
python aggregate_results.py eval_logs
```

Or with uv:
```bash
uv run --python 3.12 python aggregate_results.py eval_logs
```

### Check Individual Logs
```bash
cd src
cat eval_logs/exp_beam_width=2_n_per_beam=2_alpha=4.0.log
```

## How It Works

### Script Flow

```
1. Check/install GNU parallel
   └─> Auto-detects OS and package manager
   
2. Create eval_logs/ directory
   └─> Clears old logs
   
3. Create params.txt with experiment configs
   
4. Run all experiments in parallel
   └─> Max 4 concurrent jobs
   └─> Each saves to eval_logs/exp_*.log
   
5. Wait for all to complete
   
6. Aggregate results
   └─> Try Python script (aggregate_results.py)
   └─> Fallback to AWK if Python unavailable
   
7. Display unified results
   └─> Single comparison table
   └─> Best configuration
   └─> Improvement metrics
```

### Aggregation Logic

```python
# aggregate_results.py

1. Scan eval_logs/ for *.log files

2. For each log file:
   - Parse benchmark result tables
   - Extract: strategy, pass rate, time, tokens, cost
   
3. Combine all results:
   - Create unified table
   - Sort by experiment name
   
4. Analyze results:
   - Find best beam search config
   - Average greedy baseline
   - Calculate improvement
   
5. Display formatted output
```

## Benefits

✅ **Clear Overview** - See all results at once, not scattered across logs  
✅ **Easy Comparison** - Side-by-side comparison of all configurations  
✅ **Best Config Identified** - Automatically highlights winner  
✅ **Improvement Metrics** - Shows gain over baseline  
✅ **Reproducible** - Can re-run aggregation anytime  
✅ **Individual Logs Preserved** - Full details still available  

## Example Output

```bash
$ ./beam_search_evals.sh

Checking for GNU parallel...
✓ GNU parallel is already installed
GNU parallel 20251122

Running 5 experiments in parallel (max 4 at once)...
Logs will be saved to eval_logs/

[Experiments running...]

All experiments completed!

Aggregating results...

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
| beam_width=3_n_per_beam=2 | BeamSearch(α=4.0,width=3,n=2)     | 65.0%     | 2.34         | 12,456       | 1245.6     | $0.0037     | $0.0004      | 10       |
+---------------------------+------------------------------------+-----------+--------------+--------------+------------+-------------+--------------+----------+
| beam_width=3_n_per_beam=2 | Greedy                             | 42.0%     | 1.38         | 3,901        | 390.1      | $0.0012     | $0.0001      | 10       |
+---------------------------+------------------------------------+-----------+--------------+--------------+------------+-------------+--------------+----------+
...

==================================================================================================================
SUMMARY
==================================================================================================================

🏆 Best Beam Search Configuration:
   Strategy:   BeamSearch(α=4.0,width=3,n=2)
   Pass Rate:  65.0%
   Avg Time:   2.34s
   Total Cost: $0.0037

📊 Greedy Baseline (averaged across runs):
   Pass Rate:  41.0%
   Avg Time:   1.40s

📈 Improvement over Greedy: +24.0%

✅ Individual logs saved in eval_logs/
✅ To re-aggregate results: python aggregate_results.py eval_logs
```

## Files Modified

1. **`beam_search_evals.sh`**
   - Added log directory management
   - Redirects each experiment to separate log file
   - Calls Python aggregator at the end
   - Provides helpful status messages

2. **`aggregate_results.py`** (new)
   - Parses log files
   - Extracts benchmark results
   - Creates unified table
   - Identifies best configurations
   - Calculates improvements

3. **`PARALLEL_EXPERIMENTS_README.md`** (updated)
   - Documents new aggregation feature
   - Shows example output
   - Explains manual re-aggregation

## Technical Details

### Log File Naming
```
exp_{param1}_{param2}_{param3}.log

Example:
exp_beam_width=2_n_per_beam=2_alpha=4.0.log
```

### Table Parsing
```python
# Regex pattern to match table rows
pattern = r'\| ([^\|]+) \| ([^\|]+) \| ([^\|]+) \| ...'

# Extract:
- Benchmark name
- Model name
- Strategy name
- Pass rate
- Timing info
- Token counts
- Cost info
```

### Result Aggregation
```python
# Combine all results
all_results = []
for log in log_files:
    results = parse_log_file(log)
    all_results.extend(results)

# Display unified table
display_results(all_results)

# Find best
best = find_best_strategy(all_results)
```

## Future Enhancements

Possible improvements:
- [ ] Export to CSV for further analysis
- [ ] Plot comparison graphs
- [ ] Statistical significance testing
- [ ] Hyperparameter optimization suggestions
- [ ] Cost vs performance trade-off analysis
- [ ] Real-time progress updates during runs

## Summary

✅ **Updated**: `beam_search_evals.sh` with automatic result aggregation  
✅ **Created**: `aggregate_results.py` for parsing and display  
✅ **Updated**: Documentation with new features  
✅ **Tested**: Syntax validated  

**Ready to use!** Run `./beam_search_evals.sh` to see the new aggregated results format.
