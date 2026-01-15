# Beam Search Validation with Entropy Analysis

## Overview

The beam search implementation now includes a validation step that asks the model if each candidate is "on the right track" and analyzes the entropy of the response to measure confidence.

## Features

### 1. Candidate Validation
For each candidate generated during beam search, the model is asked:
```
Are we on the right track? Answer with only 'Yes' or 'No'.
```

### 2. Entropy Analysis
The validation response includes logprobs which are used to calculate:

- **Entropy**: Measures uncertainty in the model's answer
  - Calculated as: `H = -Σ p(x) * log₂(p(x))`
  - Range: 0 (completely certain) to ~2.3 bits (completely uncertain for binary choice)
  - **Low entropy (< 0.5)**: Model is very confident in its answer
  - **High entropy (> 1.5)**: Model is uncertain about the answer

- **Confidence**: Probability of the chosen token
  - Range: 0 to 1
  - Represents how strongly the model believes in the chosen "Yes" or "No"
  - **High confidence (> 0.9)**: Strong belief in the response
  - **Low confidence (< 0.5)**: Weak belief, model is hedging

### 3. Data Storage
Each `Beam` object now stores:
```python
@dataclass
class Beam:
    validation_response: Optional[str] = None      # "Yes" or "No"
    validation_entropy: Optional[float] = None     # Entropy in bits
    validation_confidence: Optional[float] = None  # Probability [0, 1]
```

## Usage

### Basic Usage
```python
from src.benchmark_runner import BeamSearchSampling
from openai import OpenAI

# Create beam search with validation enabled
beam_search = BeamSearchSampling(
    alpha=4.0,
    beam_width=2,
    n_per_beam=2,
    validate_candidates=True,  # Enable validation (default: True)
    debug=True,  # Show entropy and confidence in output
)

client = OpenAI(api_key="your-key", base_url="https://api.x.ai/v1")
client.default_model = "grok-beta"

# Generate with validation
completion, prompt_tokens, completion_tokens = beam_search.generate(
    client, 
    "Write a Python function to calculate factorial", 
    max_tokens=200
)
```

### Debug Output
When `debug=True`, you'll see output like:
```
[BeamSearch]   Candidate validation: Yes (entropy=0.123, conf=0.956) - def factorial(n):...
[BeamSearch]   Candidate validation: Yes (entropy=0.087, conf=0.972) - def fact(n):...
[BeamSearch]   Candidate validation: No (entropy=1.234, conf=0.678) - def broken_code():...
```

### Interpreting Results

#### Confident "Yes" (Good Candidate)
```
validation: Yes (entropy=0.08, conf=0.97)
```
- Low entropy = certain answer
- High confidence = strong "Yes"
- **Interpretation**: Model is confident this candidate is on the right track

#### Uncertain "Yes" (Questionable)
```
validation: Yes (entropy=1.52, conf=0.55)
```
- High entropy = uncertain
- Low confidence = weak "Yes"
- **Interpretation**: Model says "Yes" but isn't sure

#### Confident "No" (Bad Candidate)
```
validation: No (entropy=0.12, conf=0.94)
```
- Low entropy = certain answer
- High confidence = strong "No"
- **Interpretation**: Model is confident this is the wrong approach

## Use Cases

### 1. Filtering Low-Quality Candidates
```python
# Filter out candidates with uncertain validation
good_candidates = [
    beam for beam in candidate_beams 
    if beam.validation_entropy is not None 
    and beam.validation_entropy < 1.0  # Only keep confident answers
]
```

### 2. Adjusting Beam Scores
```python
# Penalize uncertain candidates
adjusted_score = beam.score() - (beam.validation_entropy or 0) * penalty_weight
```

### 3. Early Stopping
```python
# Stop if we get a confident "Yes"
if beam.validation_response == "Yes" and beam.validation_confidence > 0.95:
    return beam  # High confidence, stop searching
```

### 4. Debugging
```python
# Identify when model is uncertain
if beam.validation_entropy > 1.5:
    print(f"Warning: High uncertainty for candidate: {beam.text[:50]}")
```

## Implementation Details

### Validation Method
```python
async def _validate_candidate_async(
    self, 
    client: AsyncOpenAI, 
    original_prompt: str, 
    candidate_text: str
) -> tuple[str, float, float]:
    """
    Returns: (response, entropy, confidence)
    - response: "Yes" or "No"
    - entropy: entropy of logprobs (higher = more uncertain)
    - confidence: probability of chosen token (higher = more confident)
    """
```

### Entropy Calculation
1. Request `logprobs=True` and `top_logprobs=5` from the API
2. Extract probabilities: `p = exp(logprob)` for each top token
3. Normalize probabilities: `p_norm = p / sum(all_p)`
4. Calculate entropy: `H = -Σ p_norm * log₂(p_norm)`

### Performance
- Validations run **in parallel** for all candidates
- Uses `asyncio.gather()` to minimize latency
- Minimal overhead when `validate_candidates=False`

## API Requirements

Requires an API that supports:
- Chat completions endpoint
- `logprobs=True` parameter
- `top_logprobs` parameter
- Returns logprobs in response

Compatible with:
- OpenAI API
- xAI Grok API
- vLLM servers
- Most OpenAI-compatible APIs

## Testing

Run the test script:
```bash
python test_beam_validation.py
```

Set environment variables:
```bash
export XAI_API_KEY="your-api-key"
export API_BASE_URL="https://api.x.ai/v1"
export MODEL_NAME="grok-beta"
```

## Future Enhancements

Potential improvements:
1. **Adaptive filtering**: Automatically filter candidates based on entropy threshold
2. **Score adjustment**: Incorporate validation confidence into beam scoring
3. **Multi-turn validation**: Ask follow-up questions for uncertain cases
4. **Calibration**: Learn optimal entropy thresholds per problem type
5. **Aggregate metrics**: Track average validation entropy across all candidates

## References

- Shannon Entropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)
- Uncertainty in Language Models: https://arxiv.org/abs/2302.09664
- Beam Search: "Reasoning with Sampling" (https://arxiv.org/abs/2510.14901)
