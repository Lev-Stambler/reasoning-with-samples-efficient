#!/usr/bin/env python3
"""
Hydra-based script for comparing different sampling strategies on various benchmarks.
Outputs a comprehensive comparison table.

Usage:
    python run_benchmark.py                           # Run with default config
    python run_benchmark.py mcmc.steps=5              # Override MCMC steps
    python run_benchmark.py mcmc.alpha=2.0            # Override MCMC alpha
    python run_benchmark.py benchmark.num_problems=20 # More problems
    python run_benchmark.py model.name=grok-3         # Different model

Multirun (batch comparisons):
    python run_benchmark.py -m benchmark.name=humaneval,swebench
    python run_benchmark.py -m mcmc.alpha=1.0,1.67,4.0
"""
import os
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from tabulate import tabulate
from openai import OpenAI
from dataclasses import asdict
from strategies import (
    GreedySampling,
    MCMCSampling,
    ParallelMCMCSampling,
    BeamSearchSampling,
    TemperatureSampling,
    BenchmarkMetrics,
    SamplingResult,
    SamplingStrategy
)
from benchmarks import (
    Benchmark,
    HumanEvalBenchmark
)
from swebench_benchmark import SWEBenchLiteBenchmark, SWEBenchVerifiedBenchmark
from gsm8k_benchmark import GSM8KBenchmark, GSM8KTrainBenchmark
from mmlu_benchmark import (
    MMLUBenchmark,
    MMLUSTEMBenchmark,
    MMLUHumanitiesBenchmark,
    MMLUSocialSciencesBenchmark
)
from arcagi2_benchmark import ARCAGI2Benchmark, ARCAGI2TrainingBenchmark
from math_benchmark import MATHBenchmark
from gpqa_benchmark import GPQABenchmark

# Load environment variables
load_dotenv()


class BenchmarkRunner:
    """Runner for comparing different sampling strategies on any benchmark."""
    
    def __init__(
        self,
        benchmark: Benchmark,
        model_name: str,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        output_dir: str = "predictions",
        prompt_prefix: str = "",
        prompt_suffix: str = "",
        suffix_overrides: dict[str, str] | None = None  # Strategy name -> suffix override
    ):
        self.benchmark = benchmark
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.client.default_model = model_name
        self.results: list[SamplingResult] = []
        self.output_dir = output_dir
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.suffix_overrides = suffix_overrides or {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Auto-load tokenizer for MCMC/BeamSearch strategies
        SamplingStrategy.set_tokenizer_from_model(model_name)
    
    def run_single_problem(
        self,
        problem: dict,
        strategy: SamplingStrategy,
        max_tokens: int = 512
    ) -> SamplingResult:
        """Run a single problem with a given sampling strategy."""
        # Get task ID (benchmark-specific)
        task_id = problem.get("task_id") or problem.get("id") or str(problem)

        # Format prompt using benchmark
        prompt = self.benchmark.format_prompt(problem)

        # Apply custom prefix/suffix if provided
        # Check for strategy-specific suffix override
        suffix = self.suffix_overrides.get(strategy.name, self.prompt_suffix)
        if self.prompt_prefix:
            prompt = self.prompt_prefix + prompt
        if suffix:
            prompt = prompt + suffix
        
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
        strategies: list[SamplingStrategy],
        num_problems: int = 10,
        max_tokens: int = 512,
        run_id: str = None
    ) -> dict[str, BenchmarkMetrics]:
        """
        Run benchmark for multiple strategies.
        Generates prediction files for official evaluation.
        Returns: Dict mapping strategy name to metrics.
        """
        # Load benchmark dataset
        print(f"Loading {self.benchmark.name()} dataset...")
        self.benchmark.load_dataset()
        
        results_by_strategy: dict[str, list[SamplingResult]] = {s.name: [] for s in strategies}
        predictions_by_strategy: dict[str, list[dict]] = {s.name: [] for s in strategies}
        
        print(f"\nRunning benchmark on {num_problems} {self.benchmark.name()} problems...")
        print(f"Model: {self.model_name}")
        print(f"Strategies: {[s.name for s in strategies]}\n")
        
        num_problems = min(num_problems, self.benchmark.get_num_problems())
        
        for i in range(num_problems):
            problem = self.benchmark.get_problem(i)
            task_id = problem.get("task_id") or problem.get("instance_id") or problem.get("id") or f"Problem {i+1}"
            print(f"\nProblem {i+1}/{num_problems}: {task_id}")
            
            for strategy in strategies:
                print(f"  Testing {strategy.name}...", end=" ")
                try:
                    result = self.run_single_problem(problem, strategy, max_tokens)
                    results_by_strategy[strategy.name].append(result)
                    
                    # Format prediction for official evaluator
                    prediction = self.benchmark.format_prediction(problem, result.completion)
                    predictions_by_strategy[strategy.name].append(prediction)
                    
                    status = "✓ PASS" if result.passed else "✗ FAIL"
                    print(f"{status} ({result.time_seconds:.2f}s, {result.total_tokens} tokens)")
                    if not result.passed and result.metadata:
                        print(f"    {result.metadata.get('result_message', '')}")
                except Exception as e:
                    print(f"✗ ERROR: {str(e)[:50]}")
        
        # Save prediction files
        for strategy_name, predictions in predictions_by_strategy.items():
            if predictions:
                self.save_predictions(predictions, strategy_name, run_id)
        
        # Aggregate metrics
        metrics = {}
        for strategy_name, results in results_by_strategy.items():
            if not results:
                continue

            # Calculate pass rate from results
            num_passed = sum(1 for r in results if r.passed)
            pass_rate = (num_passed / len(results)) * 100.0 if results else 0.0

            metrics[strategy_name] = BenchmarkMetrics(
                model_name=self.model_name,
                strategy_name=strategy_name,
                benchmark_name=self.benchmark.name(),
                pass_rate=pass_rate,
                avg_time=sum(r.time_seconds for r in results) / len(results),
                total_tokens=sum(r.total_tokens for r in results),
                avg_tokens_per_problem=sum(r.total_tokens for r in results) / len(results),
                num_problems=len(results)
            )
        
        return metrics
    
    def save_predictions(self, predictions: list[dict], strategy_name: str, run_id: str = None):
        """Save predictions to file for official evaluation."""
        # Clean strategy name for filename
        safe_strategy = strategy_name.replace("(", "_").replace(")", "").replace("=", "").replace(",", "_").replace(" ", "")
        safe_model = self.model_name.replace("/", "_").replace("-", "_")
        safe_benchmark = self.benchmark.name().replace("-", "_").lower()
        
        # Create filename
        if run_id:
            filename = f"{safe_benchmark}_{safe_model}_{safe_strategy}_{run_id}.jsonl"
        else:
            filename = f"{safe_benchmark}_{safe_model}_{safe_strategy}.jsonl"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Write JSONL file
        with open(filepath, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        
        print(f"\n📁 Saved predictions to: {filepath}")
        print(f"   Total predictions: {len(predictions)}")
        
        # Print evaluation command
        if self.benchmark.name() == "HumanEval":
            print(f"\n   To evaluate, run:")
            print(f"   evaluate_functional_correctness {filepath}")
        elif "SWE-bench" in self.benchmark.name():
            print(f"\n   To evaluate, run:")
            print(f"   python -m swebench.harness.run_evaluation \\")
            print(f"     --predictions_path {filepath} \\")
            print(f"     --swe_bench_tasks <path-to-tasks> \\")
            print(f"     --log_dir logs/")
        
        return filepath
    
    def save_results(self, filename: str):
        """Save detailed results to JSON."""
        with open(filename, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)


# Registry of available benchmarks
BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    "swebench": SWEBenchLiteBenchmark,  # SWE-bench Lite (300 problems)
    "swebench-verified": SWEBenchVerifiedBenchmark,  # SWE-bench Verified (500 problems)
    "gsm8k": GSM8KBenchmark,  # GSM8K test set (1,319 problems)
    "gsm8k-train": GSM8KTrainBenchmark,  # GSM8K training set (7,473 problems)
    "mmlu": MMLUBenchmark,  # MMLU all subjects (~14,000 problems)
    "mmlu-stem": MMLUSTEMBenchmark,  # MMLU STEM subjects only
    "mmlu-humanities": MMLUHumanitiesBenchmark,  # MMLU Humanities subjects only
    "mmlu-social": MMLUSocialSciencesBenchmark,  # MMLU Social Sciences subjects only
    "arcagi2": ARCAGI2Benchmark,  # ARC-AGI-2 evaluation set (120 tasks)
    "arcagi2-train": ARCAGI2TrainingBenchmark,  # ARC-AGI-2 training set (1,000 tasks)
    "math": MATHBenchmark,  # MATH500 competition math problems (500 problems)
    "gpqa": GPQABenchmark,  # GPQA diamond graduate-level questions (~198 problems)
}


def print_results_table(metrics_list: list[BenchmarkMetrics]):
    """Print a formatted table of benchmark results."""

    headers = [
        "Benchmark",
        "Model",
        "Strategy",
        "Pass Rate (%)",
        "Avg Time (s)",
        "Total Tokens",
        "Avg Tokens/Problem",
        "Problems"
    ]

    rows = []
    for metrics in metrics_list:
        rows.append([
            metrics.benchmark_name,
            metrics.model_name,
            metrics.strategy_name,
            f"{metrics.pass_rate:.1f}%",
            f"{metrics.avg_time:.2f}",
            f"{metrics.total_tokens:,}",
            f"{metrics.avg_tokens_per_problem:.1f}",
            metrics.num_problems
        ])

    rows.sort(key=lambda x: float(x[3].rstrip('%')), reverse=True)

    benchmark_name = metrics_list[0].benchmark_name if metrics_list else "BENCHMARK"
    print("\n" + "="*110)
    print(f"{benchmark_name.upper()} BENCHMARK RESULTS")
    print("="*110)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("="*100)

    if rows:
        best = rows[0]
        print(f"\n🏆 Best Overall: {best[2]} on {best[0]} with {best[1]}")
        print(f"   Pass Rate: {best[3]} | Time: {best[4]}s")

        # Find best by benchmark
        benchmarks = set(m.benchmark_name for m in metrics_list)
        if len(benchmarks) > 1:
            print(f"\n📊 Best by Benchmark:")
            for bench in benchmarks:
                bench_metrics = [m for m in metrics_list if m.benchmark_name == bench]
                if bench_metrics:
                    best_bench = max(bench_metrics, key=lambda x: x.pass_rate)
                    print(f"   {bench}: {best_bench.strategy_name} ({best_bench.model_name}) - {best_bench.pass_rate:.1f}%")

        # Find best by model
        models = set(m.model_name for m in metrics_list)
        if len(models) > 1:
            print(f"\n🤖 Best by Model:")
            for model in models:
                model_metrics = [m for m in metrics_list if m.model_name == model]
                if model_metrics:
                    best_model = max(model_metrics, key=lambda x: x.pass_rate)
                    print(f"   {model}: {best_model.strategy_name} on {best_model.benchmark_name} - {best_model.pass_rate:.1f}%")


def print_summary(metrics_list: list[BenchmarkMetrics]):
    """Print a summary of key insights."""
    if not metrics_list:
        return

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    print("\nPass Rate Comparison:")
    for metrics in sorted(metrics_list, key=lambda x: x.pass_rate, reverse=True):
        bar_length = int(metrics.pass_rate / 2)
        bar = "#" * bar_length
        print(f"  {metrics.strategy_name:30s} {bar} {metrics.pass_rate:.1f}%")

    print("\nTime Efficiency:")
    for metrics in sorted(metrics_list, key=lambda x: x.avg_time):
        print(f"  {metrics.strategy_name:30s} {metrics.avg_time:.2f}s avg per problem")

    print("\nToken Usage:")
    for metrics in sorted(metrics_list, key=lambda x: x.avg_tokens_per_problem):
        print(f"  {metrics.strategy_name:30s} {metrics.avg_tokens_per_problem:.0f} tokens avg per problem")

    print("="*100 + "\n")


def print_config(cfg: DictConfig):
    """Print the current configuration."""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(OmegaConf.to_yaml(cfg))
    print("="*60 + "\n")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration."""

    if cfg.output.verbose:
        print_config(cfg)

    # Set random seed for reproducibility
    if cfg.benchmark.get("seed") is not None:
        seed = cfg.benchmark.seed
        random.seed(seed)
        np.random.seed(seed)
        print(f"🎲 Random seed set to: {seed}\n")

    # Select model config based on provider
    provider = cfg.model.get("provider", "xai")
    if provider == "vllm":
        model_name = cfg.model.vllm.name
        base_url = cfg.model.vllm.base_url
        api_key = "vllm"  # vLLM doesn't require API key
        supports_n_param = True  # vLLM supports n parameter for batching
        print(f"Using vLLM: {model_name} at {base_url}")
    elif provider == "runpod":
        model_name = cfg.model.runpod.name
        base_url = cfg.model.runpod.base_url or os.getenv("RUNPOD_ENDPOINT")
        api_key = "runpod"  # RunPod vLLM doesn't require API key
        supports_n_param = True  # RunPod vLLM supports n parameter for batching
        if not base_url:
            print("Error: Please set RUNPOD_ENDPOINT environment variable")
            print("   Or set model.runpod.base_url in config")
            return
        print(f"Using RunPod: {model_name} at {base_url}")
    else:  # xai (default)
        model_name = cfg.model.xai.name
        base_url = cfg.model.xai.base_url
        api_key = os.getenv("XAI_API_KEY")
        supports_n_param = True  # X.AI supports n parameter
        if not api_key:
            print("Error: Please set XAI_API_KEY environment variable")
            print("   Get your API key from: https://console.x.ai/")
            return

    # Initialize benchmark
    if cfg.benchmark.name not in BENCHMARK_REGISTRY:
        print(f"❌ Error: Unknown benchmark '{cfg.benchmark.name}'")
        print(f"   Available: {', '.join(BENCHMARK_REGISTRY.keys())}")
        return

    benchmark_class = BENCHMARK_REGISTRY[cfg.benchmark.name]
    benchmark = benchmark_class()

    # Build suffix overrides for strategies that have custom suffixes
    suffix_overrides = {}
    if cfg.greedy.get("suffix") is not None:
        suffix_overrides["Greedy"] = cfg.greedy.suffix

    # Initialize runner
    runner = BenchmarkRunner(
        benchmark=benchmark,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        output_dir="predictions",
        prompt_prefix=cfg.prompt.prefix,
        prompt_suffix=cfg.prompt.suffix,
        suffix_overrides=suffix_overrides
    )

    # Setup strategies based on config
    strategies = []
    # Get seed for API reproducibility (None if not set)
    api_seed = cfg.benchmark.get("seed")

    if getattr(cfg.greedy, 'enabled', False):
        strategies.append(GreedySampling(seed=api_seed))

    if getattr(cfg.mcmc, 'enabled', False):
        strategies.append(MCMCSampling(
            alpha=cfg.mcmc.alpha,
            mcmc_steps=cfg.mcmc.steps,
            top_logprobs=cfg.mcmc.top_logprobs,
            proposal_temperature=cfg.mcmc.proposal_temperature,
            restrict_to_last_n=cfg.mcmc.restrict_to_last_n,
            block_size=cfg.mcmc.block_size,
            debug=cfg.mcmc.debug,
            seed=api_seed,
        ))

    if getattr(cfg.mcmc_parallel, 'enabled', False):
        strategies.append(ParallelMCMCSampling(
            alpha=cfg.mcmc_parallel.alpha,
            mcmc_steps=cfg.mcmc_parallel.steps,
            top_logprobs=cfg.mcmc_parallel.top_logprobs,
            proposal_temperature=cfg.mcmc_parallel.proposal_temperature,
            block_size=cfg.mcmc_parallel.block_size,
            debug=cfg.mcmc_parallel.debug,
            num_proposals=cfg.mcmc_parallel.num_proposals,
            max_concurrent=cfg.mcmc_parallel.max_concurrent,
            timeout=cfg.mcmc_parallel.timeout,
            max_retries=cfg.mcmc_parallel.max_retries,
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            supports_n_param=supports_n_param,
            seed=api_seed,
            use_length_penalty=cfg.mcmc_parallel.use_length_penalty,
            length_penalty=cfg.mcmc_parallel.length_penalty,
        ))

    if getattr(cfg.beam_search, 'enabled', False):
        strategies.append(BeamSearchSampling(
            alpha=cfg.beam_search.alpha,
            beam_width=cfg.beam_search.beam_width,
            n_per_beam=cfg.beam_search.n_per_beam,
            tokens_per_step=cfg.beam_search.tokens_per_step,
            use_length_penalty=cfg.beam_search.use_length_penalty,
            length_penalty=cfg.beam_search.length_penalty,
            proposal_temperature=cfg.beam_search.proposal_temperature,
            top_logprobs=cfg.beam_search.top_logprobs,
            debug=cfg.beam_search.debug,
            supports_n_param=supports_n_param,
            max_concurrent=getattr(cfg.beam_search, 'max_concurrent', 100),
            timeout=getattr(cfg.beam_search, 'timeout', 300.0),
            seed=api_seed,
        ))

    if getattr(cfg.temperature_sampling, 'enabled', False):
        strategies.append(TemperatureSampling(
            temperature=cfg.temperature_sampling.temperature,
            seed=api_seed,
        ))

    if not strategies:
        print("❌ Error: No strategies enabled in configuration")
        return

    # Run benchmark
    try:
        metrics_dict = runner.run_benchmark(
            strategies=strategies,
            num_problems=cfg.benchmark.num_problems,
            max_tokens=cfg.benchmark.max_tokens
        )

        metrics_list = list(metrics_dict.values())

        # Display results
        print_results_table(metrics_list)
        print_summary(metrics_list)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nError during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
