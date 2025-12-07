# Length Penalty Experiments

## Overview

Updated `beam_search_evals.sh` to test different `length_penalty` values to find the optimal normalization factor for beam search scoring.

## What is Length Penalty?

Length penalty is used in beam search scoring to normalize sequence probabilities by length:

```python
score = sum(log_probs) / (length ** length_penalty)
```

**Effect of different values:**
- `length_penalty = 0.0` → No normalization (favors short sequences)
- `length_penalty = 0.6` → Moderate normalization (Google NMT default)
- `length_penalty = 0.8` → Stronger normalization
- `length_penalty = 1.0` → Full normalization (average log probability)

## Experiments

The script now tests 6 configurations:

| Experiment | beam_width | n_per_beam | alpha | length_penalty | Purpose |
|------------|-----------|------------|-------|----------------|---------|
| 1 | 2 | 2 | 4.0 | **0.6** | Baseline (Google NMT) |
| 2 | 2 | 2 | 4.0 | **0.8** | Stronger normalization |
| 3 | 2 | 2 | 4.0 | **1.0** | Full normalization |
| 4 | 3 | 2 | 4.0 | 0.6 | Larger beam width |
| 5 | 2 | 3 | 4.0 | 0.6 | More continuations/beam |
| 6 | 5 | 3 | 4.0 | 0.6 | Large beam + continuations |

## Experiments 1-3: Length Penalty Comparison

**Configuration:** Same beam search parameters, different length penalties

- **Experiment 1:** `length_penalty=0.6` (default)
  - Standard normalization
  - Used in Google Neural Machine Translation
  
- **Experiment 2:** `length_penalty=0.8`
  - Stronger length normalization
  - Less bias toward short sequences
  
- **Experiment 3:** `length_penalty=1.0`
  - Full length normalization
  - Equivalent to average log probability
  - Most aggressive de-biasing

### Expected Behavior

**Low penalty (0.6):**
- May prefer shorter, concise answers
- Higher scores for brief solutions
- Risk: Incomplete reasoning chains

**Medium penalty (0.8):**
- Balanced approach
- Moderate length preference
- Good middle ground

**High penalty (1.0):**
- Length-neutral scoring
- May prefer longer sequences
- Risk: Verbose or redundant answers

## Running the Experiments

```bash
cd src
./beam_search_evals.sh
```

This will:
1. Run all 6 experiments in parallel (max 4 concurrent)
2. Save logs to `eval_logs/exp_beam_width=*_n_per_beam=*_alpha=*_length_penalty=*.log`
3. Aggregate results automatically
4. Show comparison table with best configuration

## Analyzing Results

### Expected Output

```
==================================================================================================================
AGGREGATED BENCHMARK RESULTS FROM ALL EXPERIMENTS
==================================================================================================================

+----------------------------------------------+--------------------------------+-----------+--------------+
| Experiment                                   | Strategy                       | Pass Rate | Avg Time (s) |
+==============================================+================================+===========+==============+
| beam_width=2_n_per_beam=2_alpha=4.0_lp=0.6  | BeamSearch(α=4.0,w=2,n=2)     | 58.0%     | 1.67         |
+----------------------------------------------+--------------------------------+-----------+--------------+
| beam_width=2_n_per_beam=2_alpha=4.0_lp=0.8  | BeamSearch(α=4.0,w=2,n=2)     | 62.0%     | 1.71         |
+----------------------------------------------+--------------------------------+-----------+--------------+
| beam_width=2_n_per_beam=2_alpha=4.0_lp=1.0  | BeamSearch(α=4.0,w=2,n=2)     | 60.0%     | 1.69         |
+----------------------------------------------+--------------------------------+-----------+--------------+
...
```

### Interpreting Results

**If `length_penalty=0.6` is best:**
- Short, concise answers work well for GSM8K
- Default setting is optimal
- No changes needed

**If `length_penalty=0.8` is best:**
- Medium normalization preferred
- Update default to 0.8
- Benefits longer reasoning chains

**If `length_penalty=1.0` is best:**
- Full normalization optimal
- GSM8K benefits from longer solutions
- Update default to 1.0

## Log File Format

Each experiment creates a log file with naming pattern:
```
exp_beam_width=X_n_per_beam=Y_alpha=Z_length_penalty=W.log
```

Examples:
- `exp_beam_width=2_n_per_beam=2_alpha=4.0_length_penalty=0.6.log`
- `exp_beam_width=2_n_per_beam=2_alpha=4.0_length_penalty=0.8.log`
- `exp_beam_width=2_n_per_beam=2_alpha=4.0_length_penalty=1.0.log`

## Re-aggregate Results

To re-run aggregation after experiments complete:

```bash
cd src
python aggregate_results.py eval_logs
```

Or with uv:
```bash
uv run --python 3.12 python aggregate_results.py eval_logs
```

## Individual Experiment Inspection

View specific experiment log:

```bash
# View experiment with length_penalty=0.8
cat eval_logs/exp_beam_width=2_n_per_beam=2_alpha=4.0_length_penalty=0.8.log

# Check for errors
grep -i error eval_logs/*.log

# Compare pass rates
grep "Pass Rate" eval_logs/*.log
```

## Mathematical Details

### Beam Score Formula

```python
# Without length penalty (biased toward short sequences)
score = sum(log_target_probs)

# With length penalty (normalized)
score = sum(log_target_probs) / (length ** length_penalty)
```

### Example Comparison

Consider two beams:

**Beam A (short):**
- Tokens: 50
- Sum log probs: -100
- Raw score: -100

**Beam B (long):**
- Tokens: 100
- Sum log probs: -180
- Raw score: -180

**Without penalty (length_penalty=0.0):**
- Beam A: -100 ✅ (wins, but might be incomplete)
- Beam B: -180

**With penalty=0.6:**
- Beam A: -100 / (50^0.6) = -100 / 10.7 = -9.35
- Beam B: -180 / (100^0.6) = -180 / 15.8 = -11.4 ✅

**With penalty=1.0 (full normalization):**
- Beam A: -100 / 50 = -2.0
- Beam B: -180 / 100 = -1.8 ✅ (wins, longer but better average)

## Recommendations

1. **Run all experiments** to get empirical data
2. **Compare pass rates** across length penalties
3. **Check reasoning quality** in predictions
4. **Update default** if better value found
5. **Document findings** for future reference

## Configuration Updates

If a different length penalty performs better, update:

**File:** `src/conf/config.yaml`

```yaml
beam_search:
  length_penalty: 0.8  # Change from 0.6 if experiments show improvement
```

## Expected Timeline

With `benchmark.num_problems=10` and `max 4 parallel jobs`:

- **Experiment duration:** ~2-3 minutes each
- **Total time:** ~4-6 minutes (parallel execution)
- **Aggregation:** ~5 seconds

## Next Steps

After reviewing results:

1. **Identify best length_penalty value**
2. **Update config.yaml default**
3. **Run larger benchmark** (100+ problems) with best config
4. **Document findings** in this file
5. **Consider testing on other benchmarks** (HumanEval, etc.)

## Notes

- All experiments use same `alpha=4.0` for fair comparison
- Greedy baseline run with each experiment for reference
- Results automatically aggregated and compared
- Individual logs preserved for detailed analysis

## Summary

✅ **6 experiments** testing different configurations  
✅ **3 length penalty values** (0.6, 0.8, 1.0)  
✅ **Automatic aggregation** of results  
✅ **Best config identification** built-in  
✅ **Easy to re-run** with different parameters  

Run `./beam_search_evals.sh` to start the experiments! 🚀
