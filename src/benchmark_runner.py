import os
import time
import json
import tempfile
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from openai import OpenAI
from datasets import load_dataset
import random
import re
import numpy as np


@dataclass
class SamplingResult:
    """Results for a single problem with a specific sampling strategy."""
    task_id: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_seconds: float
    passed: bool = False
    metadata: Optional[Dict] = None  # For benchmark-specific data


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a model/strategy combination."""
    model_name: str
    strategy_name: str
    benchmark_name: str
    pass_rate: float
    avg_time: float
    total_tokens: int
    avg_tokens_per_problem: float
    num_problems: int


class SamplingStrategy:
    """Base class for sampling strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using this strategy.
        Returns: (completion, prompt_tokens, completion_tokens)
        """
        raise NotImplementedError


class GreedySampling(SamplingStrategy):
    """Greedy decoding with temperature=0."""
    
    def __init__(self):
        super().__init__("Greedy")
    
    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )


class MCMCSampling(SamplingStrategy):
    """
    MCMC power sampling with Metropolis-Hastings acceptance and partial regeneration.

    Implements sampling from target π(x) = p(x)^α using proposal q(x) = p(x).
    Based on the paper "Reasoning with Sampling" (https://arxiv.org/abs/2510.14901).

    Algorithm:
    - Target distribution: π(x) = p(x)^α where α is specified
    - Proposal distribution: q(x) = p(x) (base model, temperature=1)
    - Partial regeneration: pick random position, regenerate suffix
    - Accept/reject using MH ratio on the suffix

    For α=4: proposals with higher log probability are 3x more likely to be accepted.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        mcmc_steps: int = 10,
        top_logprobs: int = 5,
        proposal_temperature: float = 1.0,
        temperature: float = None,  # Legacy alias for proposal_temperature
        restrict_to_last_n: int = None,  # Only resample last N tokens (None = disabled)
    ):
        name = f"MCMC(α={alpha},steps={mcmc_steps})"
        if restrict_to_last_n is not None:
            name += f",lastN={restrict_to_last_n}"
        super().__init__(name)
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.top_logprobs = top_logprobs
        # Support legacy 'temperature' parameter as alias for proposal_temperature
        self.proposal_temperature = temperature if temperature is not None else proposal_temperature
        self.restrict_to_last_n = restrict_to_last_n

    def _extract_logprobs_with_tokens(self, response) -> tuple[list[str], list[float], list[float]]:
        """
        Extract tokens and logprobs from API response.

        Returns:
            (tokens, log_p, log_target)
        """
        if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
            return [], [], []

        tokens = [token.token for token in response.choices[0].logprobs.content]
        log_p = [token.logprob for token in response.choices[0].logprobs.content]
        log_target = [self.alpha * lp for lp in log_p]

        return tokens, log_p, log_target

    def _sample_full(self, client: OpenAI, prompt: str, max_tokens: int):
        """Generate a full sample from base model."""
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        text = response.choices[0].message.content
        tokens, log_p, log_target = self._extract_logprobs_with_tokens(response)

        return (
            text, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )

    def _sample_continuation(self, client: OpenAI, prompt: str, prefix: str, max_tokens: int):
        """
        Generate a continuation from a prefix using partial regeneration.

        Sends the prefix as an assistant message and lets the model continue.
        """
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}  # Continue from here
            ],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        continuation = response.choices[0].message.content
        tokens, log_p, log_target = self._extract_logprobs_with_tokens(response)

        return (
            continuation, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using MCMC power sampling with partial regeneration.

        For each MCMC step:
        1. Pick random token position idx
        2. Keep prefix (tokens 0 to idx-1)
        3. Regenerate suffix from idx onwards
        4. Accept/reject based on suffix log probabilities
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        attempts = 0
        acceptances = 0

        # Initial sample from base model
        current_text, tokens_cur, log_p_cur, log_target_cur, pt, ct = self._sample_full(
            client, prompt, max_tokens
        )
        total_prompt_tokens += pt
        total_completion_tokens += ct

        # MCMC refinement steps with partial regeneration
        for _ in range(self.mcmc_steps):
            if len(tokens_cur) < 2:
                # Need at least 2 tokens to do partial regeneration
                continue

            attempts += 1

            # Pick random position to regenerate from
            # If restrict_to_last_n is set, only resample from last N tokens
            if self.restrict_to_last_n is not None:
                min_idx = max(1, len(tokens_cur) - self.restrict_to_last_n)
            else:
                min_idx = 1  # At least keep first token
            idx = random.randint(min_idx, len(tokens_cur) - 1)

            # Prefix to keep (as text)
            prefix = "".join(tokens_cur[:idx])

            # Generate new suffix
            new_suffix, tokens_prop, log_p_prop, log_target_prop, pt, ct = self._sample_continuation(
                client, prompt, prefix, max_tokens - idx
            )
            total_prompt_tokens += pt
            total_completion_tokens += ct

            # Current suffix logprobs (from idx onwards)
            log_p_cur_suffix = log_p_cur[idx:]
            log_target_cur_suffix = log_target_cur[idx:]

            # MH acceptance ratio for suffixes only
            # log A = log(π(suffix')/π(suffix)) + log(q(suffix)/q(suffix'))
            log_r = (
                sum(log_target_prop) + sum(log_p_cur_suffix)
                - sum(log_target_cur_suffix) - sum(log_p_prop)
            )

            # Accept with probability min(1, exp(log_r))
            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                # Update current state with new suffix
                current_text = prefix + new_suffix
                tokens_cur = tokens_cur[:idx] + tokens_prop
                log_p_cur = log_p_cur[:idx] + log_p_prop
                log_target_cur = log_target_cur[:idx] + log_target_prop

        # Store acceptance ratio for diagnostics
        self._last_acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0

        return current_text, total_prompt_tokens, total_completion_tokens

    def get_acceptance_ratio(self) -> float:
        """Return the acceptance ratio from the last generate() call."""
        return getattr(self, '_last_acceptance_ratio', 0.0)


class TemperatureSampling(SamplingStrategy):
    """Standard temperature sampling."""
    
    def __init__(self, temperature: float = 0.8):
        super().__init__(f"Temperature(T={temperature})")
        self.temperature = temperature
    
    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    @abstractmethod
    def name(self) -> str:
        """Return the name of the benchmark."""
        pass
    
    @abstractmethod
    def load_dataset(self):
        """Load the benchmark dataset."""
        pass
    
    @abstractmethod
    def get_problem(self, index: int) -> Dict:
        """Get a problem by index."""
        pass
    
    @abstractmethod
    def get_num_problems(self) -> int:
        """Return total number of problems in the dataset."""
        pass
    
    @abstractmethod
    def format_prompt(self, problem: Dict) -> str:
        """Format a problem into a prompt for the LLM."""
        pass
    
    @abstractmethod
    def extract_completion(self, response: str, problem: Dict) -> str:
        """Extract the completion from LLM response."""
        pass
    
    @abstractmethod
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """
        Check if the completion is correct.
        Returns: (passed, result_message)
        """
        pass


class HumanEvalBenchmark(Benchmark):
    """HumanEval benchmark implementation."""
    
    def __init__(self):
        self.dataset = None
    
    def name(self) -> str:
        return "HumanEval"
    
    def load_dataset(self):
        if self.dataset is None:
            self.dataset = load_dataset("openai/openai_humaneval", split="test")
    
    def get_problem(self, index: int) -> Dict:
        return self.dataset[index]
    
    def get_num_problems(self) -> int:
        return len(self.dataset)
    
    def format_prompt(self, problem: Dict) -> str:
        """For HumanEval, the prompt is already in the problem."""
        return problem["prompt"]
    
    def extract_completion(self, response: str, problem: Dict) -> str:
        """Extract code completion from LLM response."""
        return extract_code_completion(response, problem["entry_point"])
    
    def check_correctness(self, problem: Dict, completion: str) -> tuple[bool, str]:
        """Check if code passes tests."""
        return check_code_execution(problem, completion)


def extract_code_completion(response: str, entry_point: str) -> str:
    """Extract code completion from LLM response."""
    # Try to find code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks, look for function definition
    lines = response.split('\n')
    code_lines = []
    in_function = False
    
    for line in lines:
        if f'def {entry_point}' in line:
            in_function = True
        if in_function:
            code_lines.append(line)
            # Stop at next function or class definition
            if line.strip().startswith('def ') and f'def {entry_point}' not in line:
                break
            if line.strip().startswith('class '):
                break
    
    if code_lines:
        return '\n'.join(code_lines).strip()
    
    # Fallback: return the whole response
    return response.strip()


def check_code_execution(problem: Dict, completion: str, timeout: float = 3.0) -> tuple[bool, str]:
    """
    Simple code execution checker.
    Returns: (passed, result_message)
    """
    check_program = (
        problem["prompt"]
        + "\n"
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})"
    )
    
    try:
        exec_globals = {}
        exec(check_program, exec_globals)
        return True, "passed"
    except Exception as e:
        return False, f"failed: {str(e)[:100]}"


class BenchmarkRunner:
    """Runner for comparing different sampling strategies on any benchmark."""
    
    def __init__(
        self,
        benchmark: Benchmark,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.x.ai/v1"
    ):
        self.benchmark = benchmark
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.client.default_model = model_name
        self.results: List[SamplingResult] = []
    
    def run_single_problem(
        self,
        problem: Dict,
        strategy: SamplingStrategy,
        max_tokens: int = 512
    ) -> SamplingResult:
        """Run a single problem with a given sampling strategy."""
        # Get task ID (benchmark-specific)
        task_id = problem.get("task_id") or problem.get("id") or str(problem)
        
        # Format prompt using benchmark
        prompt = self.benchmark.format_prompt(problem)
        
        # Generate completion
        start_time = time.time()
        completion, prompt_tokens, completion_tokens = strategy.generate(
            self.client, prompt, max_tokens
        )
        elapsed_time = time.time() - start_time
        
        # Extract completion using benchmark
        extracted_completion = self.benchmark.extract_completion(completion, problem)
        
        # Check correctness using benchmark
        passed, result_msg = self.benchmark.check_correctness(problem, extracted_completion)
        
        return SamplingResult(
            task_id=task_id,
            completion=extracted_completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_seconds=elapsed_time,
            passed=passed,
            metadata={"result_message": result_msg}
        )
    
    def run_benchmark(
        self,
        strategies: List[SamplingStrategy],
        num_problems: int = 10,
        max_tokens: int = 512
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Run benchmark for multiple strategies.
        Returns: Dict mapping strategy name to metrics.
        """
        # Load benchmark dataset
        print(f"Loading {self.benchmark.name()} dataset...")
        self.benchmark.load_dataset()
        
        results_by_strategy: Dict[str, List[SamplingResult]] = {s.name: [] for s in strategies}
        
        print(f"\nRunning benchmark on {num_problems} {self.benchmark.name()} problems...")
        print(f"Model: {self.model_name}")
        print(f"Strategies: {[s.name for s in strategies]}\n")
        
        num_problems = min(num_problems, self.benchmark.get_num_problems())
        
        for i in range(num_problems):
            problem = self.benchmark.get_problem(i)
            task_id = problem.get("task_id") or problem.get("id") or f"Problem {i+1}"
            print(f"\nProblem {i+1}/{num_problems}: {task_id}")
            
            for strategy in strategies:
                print(f"  Testing {strategy.name}...", end=" ")
                try:
                    result = self.run_single_problem(problem, strategy, max_tokens)
                    results_by_strategy[strategy.name].append(result)
                    status = "✓ PASS" if result.passed else "✗ FAIL"
                    print(f"{status} ({result.time_seconds:.2f}s, {result.total_tokens} tokens)")
                except Exception as e:
                    print(f"✗ ERROR: {str(e)[:50]}")
        
        # Aggregate metrics
        metrics = {}
        for strategy_name, results in results_by_strategy.items():
            if not results:
                continue
            
            metrics[strategy_name] = BenchmarkMetrics(
                model_name=self.model_name,
                strategy_name=strategy_name,
                benchmark_name=self.benchmark.name(),
                pass_rate=sum(r.passed for r in results) / len(results) * 100,
                avg_time=sum(r.time_seconds for r in results) / len(results),
                total_tokens=sum(r.total_tokens for r in results),
                avg_tokens_per_problem=sum(r.total_tokens for r in results) / len(results),
                num_problems=len(results)
            )
        
        return metrics
    
    def save_results(self, filename: str):
        """Save detailed results to JSON."""
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
