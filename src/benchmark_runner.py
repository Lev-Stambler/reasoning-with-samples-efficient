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
    """MCMC-inspired sampling with accept/reject."""
    
    def __init__(self, temperature: float = 0.8, mcmc_steps: int = 3):
        super().__init__(f"MCMC(T={temperature},steps={mcmc_steps})")
        self.temperature = temperature
        self.mcmc_steps = mcmc_steps
    
    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        # Initial sample
        initial_response = client.chat.completions.create(
            model=client.default_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        
        current_text = initial_response.choices[0].message.content
        current_logprob = sum([
            token.logprob for token in initial_response.choices[0].logprobs.content
        ]) if initial_response.choices[0].logprobs else 0
        
        total_prompt_tokens += initial_response.usage.prompt_tokens
        total_completion_tokens += initial_response.usage.completion_tokens
        
        # MCMC refinement
        for _ in range(self.mcmc_steps):
            proposal_response = client.chat.completions.create(
                model=client.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature * 1.2,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=5,
            )
            
            proposal_text = proposal_response.choices[0].message.content
            proposal_logprob = sum([
                token.logprob for token in proposal_response.choices[0].logprobs.content
            ]) if proposal_response.choices[0].logprobs else 0
            
            total_prompt_tokens += proposal_response.usage.prompt_tokens
            total_completion_tokens += proposal_response.usage.completion_tokens
            
            # Metropolis-Hastings acceptance
            log_ratio = proposal_logprob - current_logprob
            accept_prob = min(1.0, 2.718 ** log_ratio)
            
            if random.random() < accept_prob:
                current_text = proposal_text
                current_logprob = proposal_logprob
        
        return current_text, total_prompt_tokens, total_completion_tokens


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
