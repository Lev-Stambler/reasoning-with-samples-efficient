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
"""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from tabulate import tabulate
from benchmark_runner import (
    BenchmarkRunner,
    Benchmark,
    HumanEvalBenchmark,
    GreedySampling,
    MCMCSampling,
    TemperatureSampling,
    BenchmarkMetrics
)

# Load environment variables
load_dotenv()

# Registry of available benchmarks
BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    # Add more benchmarks here:
    # "swebench": SWEBenchBenchmark,
    # "mbpp": MBPPBenchmark,
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
        print(f"\nBest Strategy: {best[2]} with {best[3]} pass rate")
        print(f"   Average time: {best[4]}s | Tokens per problem: {best[6]}")


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

    # Check API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: Please set XAI_API_KEY environment variable")
        print("   Get your API key from: https://console.x.ai/")
        return

    # Initialize benchmark
    if cfg.benchmark.name not in BENCHMARK_REGISTRY:
        print(f"Error: Unknown benchmark '{cfg.benchmark.name}'")
        print(f"   Available: {', '.join(BENCHMARK_REGISTRY.keys())}")
        return

    benchmark_class = BENCHMARK_REGISTRY[cfg.benchmark.name]
    benchmark = benchmark_class()

    # Initialize runner
    runner = BenchmarkRunner(
        benchmark=benchmark,
        model_name=cfg.model.name,
        api_key=api_key,
        base_url=cfg.model.base_url,
    )

    # Setup strategies based on config
    strategies = []

    if cfg.greedy.enabled:
        strategies.append(GreedySampling())

    if cfg.mcmc.enabled:
        strategies.append(MCMCSampling(
            alpha=cfg.mcmc.alpha,
            mcmc_steps=cfg.mcmc.steps,
            top_logprobs=cfg.mcmc.top_logprobs,
            proposal_temperature=cfg.mcmc.proposal_temperature,
            restrict_to_last_n=cfg.mcmc.restrict_to_last_n,
        ))

    if cfg.temperature_sampling.enabled:
        strategies.append(TemperatureSampling(
            temperature=cfg.temperature_sampling.temperature
        ))

    if not strategies:
        print("Error: No strategies enabled in configuration")
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
