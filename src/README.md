# LLM Sampling Strategy Benchmark

A flexible framework for comparing different sampling strategies across various benchmarks.

## Features

- **Multiple Sampling Strategies**: Greedy, MCMC, Temperature-based sampling
- **Extensible Benchmarks**: Easy to add new benchmarks (HumanEval included, SWEBench/MATH templates provided)
- **Comprehensive Metrics**: Pass rate, time, token usage, and more
- **Beautiful Output**: Formatted tables and visualizations

## Quick Start

### 1. Setup Environment

```bash
# Make sure your API key is in .env
echo "XAI_API_KEY=your-api-key-here" > .env

# Install dependencies (if not already done)
uv pip install -e ".[dev]"
```

### 2. Run Benchmark

```bash
# Compare all strategies on HumanEval (first 10 problems)
python src/run_benchmark.py

# With custom options
python src/run_benchmark.py \
  --benchmark humaneval \
  --model grok-2-1212 \
  --num-problems 20 \
  --strategies greedy,mcmc,temp \
  --temperature 0.8 \
  --mcmc-steps 3
```

### 3. View Results

The script outputs:
- A comparison table with all metrics
- Pass rate comparison bar chart
- Time efficiency rankings
- Token usage analysis

Example output:
```
==================================================
HUMANEVAL BENCHMARK RESULTS
==================================================
┌────────────┬──────────────┬─────────────┬──────────────┬────────┬────────┐
│ Benchmark  │ Model        │ Strategy    │ Pass Rate    │ Time   │ Tokens │
├────────────┼──────────────┼─────────────┼──────────────┼────────┼────────┤
│ HumanEval  │ grok-2-1212  │ MCMC(...)   │ 75.0%        │ 12.34s │ 1,234  │
│ HumanEval  │ grok-2-1212  │ Greedy      │ 70.0%        │ 4.56s  │ 567    │
│ HumanEval  │ grok-2-1212  │ Temperature │ 65.0%        │ 5.67s  │ 678    │
└────────────┴──────────────┴─────────────┴──────────────┴────────┴────────┘
```

## Architecture

### Core Components

1. **`Benchmark` (Abstract Base Class)**: Defines interface for benchmarks
   - `load_dataset()`: Load the benchmark data
   - `format_prompt()`: Format problems for LLM
   - `extract_completion()`: Parse LLM responses
   - `check_correctness()`: Verify solutions

2. **`SamplingStrategy` (Abstract Base Class)**: Defines sampling methods
   - `generate()`: Generate completion with specific strategy

3. **`BenchmarkRunner`**: Orchestrates benchmarking
   - Works with any benchmark implementation
   - Tracks metrics (pass rate, time, tokens)
   - Handles errors gracefully

### File Structure

```
src/
├── benchmark_runner.py      # Core framework (Benchmark, Runner, Strategies)
├── benchmark_template.py    # Examples for adding new benchmarks
├── run_benchmark.py         # Main CLI script
├── llm_wrapper.py          # Custom LLM wrapper (optional)
├── test_sampling.py        # Simple test script
└── README.md               # This file
```

## Adding New Benchmarks

### Step 1: Implement Benchmark Class

Create a new class that inherits from `Benchmark`:

```python
from benchmark_runner import Benchmark

class MyBenchmark(Benchmark):
    def name(self) -> str:
        return "MyBenchmark"
    
    def load_dataset(self):
        # Load your dataset
        pass
    
    def get_problem(self, index: int) -> Dict:
        # Return problem at index
        pass
    
    def get_num_problems(self) -> int:
        # Return total problems
        pass
    
    def format_prompt(self, problem: Dict) -> str:
        # Convert problem to LLM prompt
        pass
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        # Extract answer from LLM response
        pass
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        # Verify correctness
        return True, "correct"
```

### Step 2: Register Benchmark

Add to `BENCHMARK_REGISTRY` in `run_benchmark.py`:

```python
BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    "mybenchmark": MyBenchmark,  # Add here
}
```

### Step 3: Use It

```bash
python src/run_benchmark.py --benchmark mybenchmark
```

See `benchmark_template.py` for complete examples (SWEBench, MATH).

## Available Strategies

### 1. Greedy Sampling
- Temperature = 0 (deterministic)
- Always picks most likely token
- Fast, consistent, but may miss creative solutions

### 2. MCMC Sampling
- Uses Metropolis-Hastings accept/reject
- Explores alternative solutions
- Refines through multiple proposals
- Configurable: `--mcmc-steps N`

### 3. Temperature Sampling
- Standard stochastic sampling
- Configurable: `--temperature T`
- Higher T = more diverse outputs

## Adding New Strategies

Inherit from `SamplingStrategy`:

```python
class MyStrategy(SamplingStrategy):
    def __init__(self):
        super().__init__("MyStrategy")
    
    def generate(self, client, prompt, max_tokens):
        # Your sampling logic here
        response = client.chat.completions.create(...)
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
```

## CLI Options

```
--benchmark         Benchmark to use (humaneval, etc.)
--model            LLM model name (default: grok-2-1212)
--num-problems     Number of problems to test (default: 10)
--max-tokens       Max tokens per completion (default: 512)
--strategies       Comma-separated strategies (default: greedy,mcmc,temp)
--mcmc-steps       MCMC refinement steps (default: 3)
--temperature      Sampling temperature (default: 0.8)
```

## Metrics Tracked

- **Pass Rate (%)**: Percentage of problems solved correctly
- **Avg Time (s)**: Average time per problem
- **Total Tokens**: Total tokens used across all problems
- **Avg Tokens/Problem**: Average tokens per problem

## Examples

### Compare strategies on first 5 problems
```bash
python src/run_benchmark.py --num-problems 5
```

### Test only MCMC with high temperature
```bash
python src/run_benchmark.py \
  --strategies mcmc \
  --temperature 1.0 \
  --mcmc-steps 5
```

### Use different model
```bash
python src/run_benchmark.py --model grok-beta
```

## Extending the Framework

The framework is designed to be easily extensible:

1. **New Benchmarks**: See `benchmark_template.py` for SWEBench/MATH examples
2. **New Strategies**: Inherit from `SamplingStrategy`
3. **New Metrics**: Extend `BenchmarkMetrics` dataclass
4. **New Models**: Just change `--model` parameter (works with any OpenAI-compatible API)

## Troubleshooting

**ImportError**: Make sure all dependencies are installed
```bash
uv pip install -e ".[dev]"
```

**API Key Error**: Set your API key in `.env`
```bash
echo "XAI_API_KEY=your-key" > .env
```

**Execution Errors**: Some benchmarks (like HumanEval) execute code. This is sandboxed but may fail on complex problems.

## License

MIT
