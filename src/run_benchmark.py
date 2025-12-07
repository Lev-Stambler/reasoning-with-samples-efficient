#!/usr/bin/env python3
"""
Main script for comparing different sampling strategies on various benchmarks.
Outputs a comprehensive comparison table.
"""
import os
import argparse
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
    
    # Prepare table data
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
    
    # Sort by pass rate (descending)
    rows.sort(key=lambda x: float(x[3].rstrip('%')), reverse=True)
    
    benchmark_name = metrics_list[0].benchmark_name if metrics_list else "BENCHMARK"
    print("\n" + "="*110)
    print(f"{benchmark_name.upper()} BENCHMARK RESULTS")
    print("="*110)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("="*100)
    
    # Find best performing
    if rows:
        best = rows[0]
        print(f"\n🏆 Best Strategy: {best[2]} with {best[3]} pass rate")
        print(f"   Average time: {best[4]}s | Tokens per problem: {best[6]}")


def print_summary(metrics_list: list[BenchmarkMetrics]):
    """Print a summary of key insights."""
    if not metrics_list:
        return
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    # Pass rate comparison
    print("\n📊 Pass Rate Comparison:")
    for metrics in sorted(metrics_list, key=lambda x: x.pass_rate, reverse=True):
        bar_length = int(metrics.pass_rate / 2)
        bar = "█" * bar_length
        print(f"  {metrics.strategy_name:30s} {bar} {metrics.pass_rate:.1f}%")
    
    # Time efficiency
    print("\n⚡ Time Efficiency:")
    for metrics in sorted(metrics_list, key=lambda x: x.avg_time):
        print(f"  {metrics.strategy_name:30s} {metrics.avg_time:.2f}s avg per problem")
    
    # Token usage
    print("\n🎫 Token Usage:")
    for metrics in sorted(metrics_list, key=lambda x: x.avg_tokens_per_problem):
        print(f"  {metrics.strategy_name:30s} {metrics.avg_tokens_per_problem:.0f} tokens avg per problem")
    
    print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare sampling strategies on various benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=list(BENCHMARK_REGISTRY.keys()),
        help=f"Benchmark to use (default: humaneval, available: {', '.join(BENCHMARK_REGISTRY.keys())})"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grok-2-1212",
        help="Model to use (default: grok-2-1212)"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help="Number of HumanEval problems to test (default: 10)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens per completion (default: 512)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="greedy,mcmc,temp",
        help="Comma-separated list of strategies: greedy,mcmc,temp (default: all)"
    )
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=3,
        help="Number of MCMC steps (default: 3)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling strategies (default: 0.8)"
    )
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set XAI_API_KEY environment variable")
        print("   Get your API key from: https://console.x.ai/")
        return
    
    # Initialize benchmark
    benchmark_class = BENCHMARK_REGISTRY[args.benchmark]
    benchmark = benchmark_class()
    
    # Initialize runner
    runner = BenchmarkRunner(
        benchmark=benchmark,
        model_name=args.model,
        api_key=api_key,
    )
    
    # Setup strategies
    strategies = []
    strategy_names = args.strategies.lower().split(',')
    
    if 'greedy' in strategy_names:
        strategies.append(GreedySampling())
    if 'mcmc' in strategy_names:
        strategies.append(MCMCSampling(
            temperature=args.temperature,
            mcmc_steps=args.mcmc_steps
        ))
    if 'temp' in strategy_names:
        strategies.append(TemperatureSampling(temperature=args.temperature))
    
    if not strategies:
        print("❌ Error: No valid strategies specified")
        print("   Available strategies: greedy, mcmc, temp")
        return
    
    # Run benchmark
    try:
        metrics_dict = runner.run_benchmark(
            strategies=strategies,
            num_problems=args.num_problems,
            max_tokens=args.max_tokens
        )
        
        metrics_list = list(metrics_dict.values())
        
        # Display results
        print_results_table(metrics_list)
        print_summary(metrics_list)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
