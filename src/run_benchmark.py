#!/usr/bin/env python3
"""
Main script for comparing different sampling strategies on various benchmarks.
Supports running multiple models, benchmarks, and strategies in parallel.
Outputs a comprehensive comparison table.
"""
import os
import argparse
from itertools import product
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
from swebench_benchmark import SWEBenchLiteBenchmark, SWEBenchVerifiedBenchmark

# Load environment variables
load_dotenv()

# Registry of available benchmarks
BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    "swebench": SWEBenchLiteBenchmark,  # SWE-bench Lite (300 problems)
    "swebench-verified": SWEBenchVerifiedBenchmark,  # SWE-bench Verified (500 problems)
    # Add more benchmarks here:
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
        "Total Cost ($)",
        "Cost/Problem ($)",
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
            f"${metrics.total_cost:.4f}",
            f"${metrics.cost_per_problem:.4f}",
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
        print(f"\n🏆 Best Overall: {best[2]} on {best[0]} with {best[1]}")
        print(f"   Pass Rate: {best[3]} | Time: {best[4]}s | Cost: {best[8]}")
        
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
    
    # Cost efficiency
    print("\n💰 Cost Efficiency:")
    for metrics in sorted(metrics_list, key=lambda x: x.cost_per_problem):
        print(f"  {metrics.strategy_name:30s} ${metrics.cost_per_problem:.4f} per problem (${metrics.total_cost:.4f} total)")
    
    print("="*100 + "\n")


def parse_list_arg(arg_value: str) -> list[str]:
    """Parse comma-separated argument into list."""
    return [item.strip() for item in arg_value.split(',')]


def parse_numeric_list_arg(arg_value: str) -> list[float]:
    """Parse comma-separated numeric argument into list."""
    return [float(item.strip()) for item in arg_value.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Compare sampling strategies on various benchmarks. "
                    "All arguments support comma-separated lists for batch comparison.",
        epilog="Example: --model grok-beta,grok-2-1212 --benchmark humaneval,swebench --strategies greedy,mcmc"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        help=f"Benchmark(s) to use, comma-separated (default: humaneval). "
             f"Available: {', '.join(BENCHMARK_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="grok-2-1212",
        help="Model(s) to use, comma-separated (default: grok-2-1212). "
             "Example: grok-beta,grok-2-1212"
    )
    parser.add_argument(
        "--num-problems",
        type=str,
        default="10",
        help="Number of problems to test, comma-separated for multiple runs (default: 10). "
             "Example: 5,10,20"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens per completion (default: 512 for HumanEval, 2048 for SWE-bench)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="greedy,mcmc,temp",
        help="Comma-separated list of strategies: greedy,mcmc,temp (default: all)"
    )
    parser.add_argument(
        "--mcmc-steps",
        type=str,
        default="3",
        help="Number of MCMC steps, comma-separated for multiple configurations (default: 3). "
             "Example: 2,3,5"
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.8",
        help="Temperature for sampling strategies, comma-separated (default: 0.8). "
             "Example: 0.7,0.8,0.9"
    )
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("❌ Error: Please set XAI_API_KEY environment variable")
        print("   Get your API key from: https://console.x.ai/")
        return
    
    # Parse list arguments
    benchmark_names = parse_list_arg(args.benchmark)
    model_names = parse_list_arg(args.model)
    num_problems_list = [int(x) for x in parse_list_arg(args.num_problems)]
    strategy_names = parse_list_arg(args.strategies.lower())
    mcmc_steps_list = [int(x) for x in parse_list_arg(args.mcmc_steps)]
    temperature_list = [float(x) for x in parse_list_arg(args.temperature)]
    
    # Validate benchmarks
    for bench in benchmark_names:
        if bench not in BENCHMARK_REGISTRY:
            print(f"❌ Error: Unknown benchmark '{bench}'")
            print(f"   Available: {', '.join(BENCHMARK_REGISTRY.keys())}")
            return
    
    # Validate strategies
    valid_strategies = {'greedy', 'mcmc', 'temp'}
    for strat in strategy_names:
        if strat not in valid_strategies:
            print(f"❌ Error: Unknown strategy '{strat}'")
            print(f"   Available: {', '.join(valid_strategies)}")
            return
    
    # Build strategy configurations
    # For MCMC and temp, create variants for each temperature/mcmc_steps combination
    strategy_configs = []
    
    if 'greedy' in strategy_names:
        strategy_configs.append(('greedy', None, None))
    
    if 'mcmc' in strategy_names:
        for temp in temperature_list:
            for steps in mcmc_steps_list:
                strategy_configs.append(('mcmc', temp, steps))
    
    if 'temp' in strategy_names:
        for temp in temperature_list:
            strategy_configs.append(('temp', temp, None))
    
    # Generate all combinations
    all_combinations = list(product(
        model_names,
        benchmark_names,
        num_problems_list,
        strategy_configs
    ))
    
    total_runs = len(all_combinations)
    print(f"\n{'='*80}")
    print(f"RUNNING {total_runs} BENCHMARK COMBINATIONS")
    print(f"{'='*80}")
    print(f"Models: {', '.join(model_names)}")
    print(f"Benchmarks: {', '.join(benchmark_names)}")
    print(f"Problem counts: {', '.join(map(str, num_problems_list))}")
    print(f"Strategies: {len(strategy_configs)} configurations")
    print(f"{'='*80}\n")
    
    # Collect all results
    all_metrics = []
    
    try:
        for run_idx, (model_name, benchmark_name, num_problems, (strat_type, temp, steps)) in enumerate(all_combinations, 1):
            print(f"\n{'='*80}")
            print(f"RUN {run_idx}/{total_runs}")
            print(f"Model: {model_name} | Benchmark: {benchmark_name} | Problems: {num_problems}")
            
            # Create strategy
            if strat_type == 'greedy':
                strategy = GreedySampling()
                print(f"Strategy: Greedy")
            elif strat_type == 'mcmc':
                strategy = MCMCSampling(temperature=temp, mcmc_steps=steps)
                print(f"Strategy: MCMC (temp={temp}, steps={steps})")
            elif strat_type == 'temp':
                strategy = TemperatureSampling(temperature=temp)
                print(f"Strategy: Temperature (temp={temp})")
            
            print(f"{'='*80}")
            
            # Initialize benchmark
            benchmark_class = BENCHMARK_REGISTRY[benchmark_name]
            benchmark = benchmark_class()
            
            # Initialize runner
            runner = BenchmarkRunner(
                benchmark=benchmark,
                model_name=model_name,
                api_key=api_key,
                output_dir="predictions"
            )
            
            # Set max_tokens based on benchmark if not specified
            max_tokens = args.max_tokens
            if max_tokens is None:
                if benchmark_name in ["swebench", "swebench-verified"]:
                    max_tokens = 2048
                else:
                    max_tokens = 512
            
            # Run single benchmark with run_id for unique filenames
            run_id = f"run{run_idx}"
            metrics_dict = runner.run_benchmark(
                strategies=[strategy],
                num_problems=num_problems,
                max_tokens=max_tokens,
                run_id=run_id
            )
            
            # Collect metrics
            for metrics in metrics_dict.values():
                all_metrics.append(metrics)
        
        # Display combined results
        if all_metrics:
            print(f"\n\n{'='*80}")
            print("COMBINED RESULTS - ALL RUNS")
            print(f"{'='*80}\n")
            print_results_table(all_metrics)
            print_summary(all_metrics)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        if all_metrics:
            print("\nShowing results from completed runs:")
            print_results_table(all_metrics)
    except Exception as e:
        print(f"\n❌ Error during benchmark: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
