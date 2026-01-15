# Beam Search Validation - Quick Start Guide

## TL;DR

Beam search now validates each candidate by asking the model "Are we on the right track?" and measures confidence using entropy from logprobs.

## Quick Example

```python
from src.benchmark_runner import BeamSearchSampling
from openai import OpenAI

# Setup
client = OpenAI(api_key="your-key", base_url="https://api.x.ai/v1")
client.default_model = "grok-beta"

# Create beam search with validation
beam_search = BeamSearchSampling(
    validate_candidates=True,  # Enable validation
    debug=True  # Show entropy/confidence
)

# Generate
completion, _, _ = beam_search.generate(client, "Write a factorial function", max_tokens=200)
```

## What You'll See

```
[BeamSearch]   Candidate validation: Yes (entropy=0.123, conf=0.956) - def factorial(n):...
[BeamSearch]   Candidate validation: No (entropy=0.087, conf=0.972) - def wrong_approach():...
```

## Understanding the Output

- **Response**: "Yes" = on right track, "No" = wrong approach
- **Entropy**: Lower = more certain (< 0.5 is very confident)
- **Confidence**: Higher = stronger belief (> 0.9 is strong)

## Key Files

- **`benchmark_runner.py`**: Core implementation (modified)
- **`test_beam_validation.py`**: Test script to run
- **`analyze_beam_validation.py`**: Utility functions
- **`BEAM_VALIDATION_ENTROPY.md`**: Full documentation
- **`VALIDATION_SUMMARY.md`**: Detailed summary

## Common Patterns

### Disable Validation
```python
beam_search = BeamSearchSampling(validate_candidates=False)
```

### Filter Confident "Yes" Candidates
```python
from analyze_beam_validation import filter_by_validation

good_beams = filter_by_validation(
    beams, 
    require_yes=True, 
    max_entropy=0.5, 
    min_confidence=0.9
)
```

### Rank by Validation Quality
```python
from analyze_beam_validation import rank_by_validation

ranked = rank_by_validation(beams)
best_beam = ranked[0][1]
```

## Run the Test

```bash
export XAI_API_KEY="your-key"
export API_BASE_URL="https://api.x.ai/v1"
export MODEL_NAME="grok-beta"

python test_beam_validation.py
```

## What Changed in `Beam` Class

```python
@dataclass
class Beam:
    # New fields:
    validation_response: Optional[str] = None      # "Yes" or "No"
    validation_entropy: Optional[float] = None     # 0 to ~2.3 bits
    validation_confidence: Optional[float] = None  # 0 to 1
```

## Entropy Quick Reference

| Entropy | Meaning |
|---------|---------|
| < 0.3 | Very confident |
| 0.3-0.8 | Confident |
| 0.8-1.5 | Somewhat uncertain |
| > 1.5 | Very uncertain |

## Confidence Quick Reference

| Confidence | Meaning |
|------------|---------|
| > 0.95 | Very strong belief |
| 0.85-0.95 | Strong belief |
| 0.70-0.85 | Moderate belief |
| < 0.70 | Weak belief |

## Need More Details?

See `VALIDATION_SUMMARY.md` for comprehensive documentation.
