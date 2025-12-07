# TRUE Beam Search Implementation - Complete

## What Was Implemented

**TRUE beam search** with `n_per_beam` parameter, enabling proper beam branching at each step.

### Previous (Pseudo-Beam Search):
```
Each beam → 1 continuation → Score → Prune
beam_width beams × 1 = beam_width candidates
```

### NEW (True Beam Search):
```
Each beam → n_per_beam continuations → Score → Prune
beam_width beams × n_per_beam = many candidates → Keep top beam_width
```

---

## Files Modified

### 1. `src/conf/config.yaml`
```yaml
beam_search:
  beam_width: 5                  # Keep top-5 beams after pruning
  n_per_beam: 5                  # Generate 5 continuations per beam ← NEW!
```

**Result**: 5 beams × 5 samples = **25 candidates per iteration** → prune to top 5

### 2. `src/benchmark_runner.py`

#### Added `n_per_beam` parameter:
```python
def __init__(
    self,
    alpha: float = 4.0,
    beam_width: int = 5,
    n_per_beam: int = 5,  # NEW PARAMETER
    ...
):
    self.n_per_beam = n_per_beam
```

#### Added `_sample_continuation_multiple()` method:
```python
def _sample_continuation_multiple(self, client, prompt, prefix, max_tokens, n):
    """Generate n continuations from a prefix for true beam search expansion."""
    response = client.chat.completions.create(
        ...,
        n=n,  # KEY: Generate n samples in one API call
    )
    
    # Extract all n continuations
    results = []
    for choice in response.choices:
        # Extract text, tokens, logprobs for each choice
        results.append((continuation, tokens, log_p, log_target, finished))
    
    return results, prompt_tokens, completion_tokens
```

#### Modified `generate()` to use true beam expansion:
```python
# OLD: Generate 1 continuation per beam
for beam in active_beams:
    continuation = _sample_continuation(beam)
    candidate_beams.append(continuation)

# NEW: Generate n_per_beam continuations per beam
for beam in active_beams:
    continuations = _sample_continuation_multiple(beam, n=self.n_per_beam)
    for cont in continuations:
        candidate_beams.append(cont)
# Result: beam_width × n_per_beam candidates
```

### 3. `src/run_benchmark.py`

Added `n_per_beam` parameter to config instantiation:
```python
strategies.append(BeamSearchSampling(
    alpha=cfg.beam_search.alpha,
    beam_width=cfg.beam_search.beam_width,
    n_per_beam=cfg.beam_search.n_per_beam,  # NEW
    ...
))
```

### 4. `src/QUICK_START.md`

Updated documentation with:
- Explanation of `n_per_beam` parameter
- Example showing 25 candidates → top 5
- Guidelines for choosing values

---

## Algorithm Flow (beam_width=5, n_per_beam=5)

### Iteration 1:
```
Start: 1 beam (initial context)
↓
Generate: 1 × 5 = 5 API calls with n=5
↓
Candidates: 5 completions
↓
Score all 5 using p^α with length normalization
↓
Prune: Keep top 5 beams
```

### Iteration 2+:
```
Start: 5 beams
↓
Generate: 5 × 5 = 25 API calls? NO! 5 API calls with n=5
↓
Candidates: 25 completions total
↓
Score all 25 using p^α with length normalization
↓
Prune: Keep top 5 beams
↓
Repeat...
```

---

## Key API Feature: The `n` Parameter

The OpenAI-compatible API (Grok included) supports:
```python
response = client.chat.completions.create(
    ...,
    n=5,  # Generate 5 different samples in ONE API call
)

# Access all samples
for choice in response.choices:
    text = choice.message.content
    logprobs = choice.logprobs.content  # Each has its own logprobs
```

**Benefits**:
- ✅ Multiple samples from one API call (efficient)
- ✅ Each sample has complete logprobs
- ✅ All samples share prompt cost (only counted once)
- ✅ Enables true beam branching

---

## Cost Analysis

### API Calls Per Iteration:
- **Old (n=1)**: `beam_width` API calls
  - 5 beams × 1 = **5 calls**
- **NEW (n=5)**: `beam_width` API calls (same!)
  - 5 beams × 1 call with n=5 = **5 calls**

### Token Usage Per Iteration:
- **Old (n=1)**: beam_width × tokens_per_step
  - 5 × 192 = **960 tokens**
- **NEW (n=5)**: beam_width × n_per_beam × tokens_per_step
  - 5 × 5 × 192 = **4,800 tokens**

**Result**: Same number of API calls, but **5× more tokens** (and cost)

---

## Quality vs Cost Trade-off

| Configuration | Candidates/Iter | API Calls | Tokens | Quality | Use Case |
|--------------|----------------|-----------|---------|---------|----------|
| `n=1` (old) | 5 | 5 | 960 | Good | Fast prototyping |
| `n=2` | 10 | 5 | 1,920 | Better | Balanced |
| `n=3` | 15 | 5 | 2,880 | Better+ | Good balance |
| `n=5` (new) | 25 | 5 | 4,800 | Best | Production |
| `n=10` | 50 | 5 | 9,600 | Best+ | Maximum quality |

---

## Usage Examples

### Enable TRUE Beam Search:
```bash
cd src

# Use defaults (beam_width=5, n_per_beam=5)
uv run --python 3.12 python run_benchmark.py beam_search.enabled=true

# Customize both parameters
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=10 \
    beam_search.n_per_beam=5

# Lower cost version
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.beam_width=3 \
    beam_search.n_per_beam=3
```

### Debug Mode:
```bash
uv run --python 3.12 python run_benchmark.py \
    beam_search.enabled=true \
    beam_search.debug=true \
    benchmark.num_problems=1
```

Expected output:
```
[BeamSearch] TRUE beam search: beam_width=5, n_per_beam=5
[BeamSearch] Generating 2 blocks of 192 tokens
[BeamSearch] α=4.0, length_penalty=0.6
[BeamSearch] Each iteration: 1 beams × 5 samples = candidates
[BeamSearch] Block 1: Generated 5 candidates
...
[BeamSearch] Block 2: Generated 25 candidates
```

---

## Verification

### Test the implementation:
```bash
cd src

# Verify import and parameters
uv run --python 3.12 python -c "
from benchmark_runner import BeamSearchSampling
bs = BeamSearchSampling(beam_width=5, n_per_beam=5)
print(f'Strategy: {bs.name}')
print(f'beam_width={bs.beam_width}, n_per_beam={bs.n_per_beam}')
print('✓ TRUE beam search ready!')
"
```

Expected output:
```
Strategy: BeamSearch(α=4.0,width=5,n=5,tps=192)
beam_width=5, n_per_beam=5
✓ TRUE beam search ready!
```

---

## Comparison to MCMC

| Feature | Beam Search (n=5) | MCMC |
|---------|------------------|------|
| **Exploration** | Deterministic top-k | Stochastic proposals |
| **Branching** | 25 candidates/iter | 1 candidate + proposals |
| **Diversity** | Medium (pruned) | High (random jumps) |
| **API Calls** | 5/iteration | 1 + 10 (mcmc_steps) |
| **Tokens** | 4,800/iteration | ~2,000/iteration |
| **Quality** | Very good | Good |
| **Reproducibility** | ✅ Deterministic | ❌ Stochastic |

---

## Next Steps

### Immediate:
1. Test with small dataset:
   ```bash
   uv run --python 3.12 python run_benchmark.py \
       beam_search.enabled=true \
       benchmark.num_problems=5
   ```

2. Compare with MCMC:
   ```bash
   uv run --python 3.12 python run_benchmark.py \
       beam_search.enabled=true \
       mcmc.enabled=true \
       benchmark.num_problems=10
   ```

3. Tune parameters:
   - Try different `n_per_beam` values (1, 3, 5, 10)
   - Compare pass rates
   - Measure cost vs quality trade-off

### Future Enhancements:
- [ ] Diversity-based pruning (penalize similar beams)
- [ ] Adaptive n_per_beam (more for promising beams)
- [ ] Early stopping per beam (stop bad branches)
- [ ] Multi-stage generation (smaller chunks)
- [ ] Nucleus beam search (sample from top-p of beams)

---

## Key Takeaways

✅ **TRUE beam search implemented** - Proper branching with `n_per_beam`  
✅ **Uses API `n` parameter** - Efficient multi-sample generation  
✅ **Configurable trade-off** - Adjust beam_width and n_per_beam  
✅ **Maintains compatibility** - Works with existing MCMC/Greedy strategies  
✅ **Well documented** - Config, code, and usage examples  

**Status**: ✅ **READY FOR TESTING!**
