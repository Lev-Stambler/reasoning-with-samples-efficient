import os
import random
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file
load_dotenv()

# Model configuration
GROK_MODEL = "grok-2-1212"  # Update to "grok-beta" or other model as needed


def greedy_decode(client, prompt, max_tokens=512):
    """Greedy decoding: temperature=0 for deterministic output."""
    response = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def mcmc_sampling(client, prompt, max_tokens=512, alpha=4.0, mcmc_steps=10):
    """
    MCMC power sampling with Metropolis-Hastings acceptance and partial regeneration.

    Implements sampling from target π(x) = p(x)^α using proposal q(x) = p(x).
    Based on the paper "Reasoning with Sampling" (https://arxiv.org/abs/2510.14901).

    Algorithm:
    - Target: π = p^α (power distribution)
    - Proposal: q = p (base model, temperature=1)
    - Partial regeneration: pick random position, regenerate suffix
    - Accept/reject based on suffix log probabilities

    For α=4: proposals with higher log probability are 3x more likely to be accepted.

    Args:
        client: OpenAI client
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        alpha: Power for target distribution p^α (default 4.0)
        mcmc_steps: Number of MCMC refinement steps (default 10)

    Returns:
        tuple: (text, log_p, log_target, acceptance_ratio)
    """
    attempts = 0
    acceptances = 0

    def extract_logprobs_with_tokens(response):
        """Extract tokens, log p, and log target."""
        if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
            return [], [], []

        tokens = [t.token for t in response.choices[0].logprobs.content]
        log_p = [t.logprob for t in response.choices[0].logprobs.content]
        log_target = [alpha * lp for lp in log_p]

        return tokens, log_p, log_target

    # Initial sample from base model
    initial_response = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=5,
    )

    current_text = initial_response.choices[0].message.content
    tokens_cur, log_p_cur, log_target_cur = extract_logprobs_with_tokens(initial_response)

    # MCMC refinement steps with partial regeneration
    for step in range(mcmc_steps):
        if len(tokens_cur) < 2:
            continue

        attempts += 1

        # Pick random position to regenerate from
        idx = random.randint(1, len(tokens_cur) - 1)
        prefix = "".join(tokens_cur[:idx])

        # Generate new suffix by continuing from prefix
        proposal_response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}  # Continue from here
            ],
            temperature=1.0,
            max_tokens=max_tokens - idx,
            logprobs=True,
            top_logprobs=5,
        )

        new_suffix = proposal_response.choices[0].message.content
        tokens_prop, log_p_prop, log_target_prop = extract_logprobs_with_tokens(proposal_response)

        # Current suffix logprobs
        log_p_cur_suffix = log_p_cur[idx:]
        log_target_cur_suffix = log_target_cur[idx:]

        # MH acceptance ratio for suffixes
        log_r = (
            sum(log_target_prop) + sum(log_p_cur_suffix)
            - sum(log_target_cur_suffix) - sum(log_p_prop)
        )

        if np.random.rand() < np.exp(log_r):
            acceptances += 1
            current_text = prefix + new_suffix
            tokens_cur = tokens_cur[:idx] + tokens_prop
            log_p_cur = log_p_cur[:idx] + log_p_prop
            log_target_cur = log_target_cur[:idx] + log_target_prop
            print(f"  [Step {step+1}] Accepted at idx={idx} (log_r: {log_r:.3f})")
        else:
            print(f"  [Step {step+1}] Rejected at idx={idx} (log_r: {log_r:.3f})")

    acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0
    print(f"  Acceptance ratio: {acceptance_ratio:.2%}")

    return current_text, log_p_cur, log_target_cur, acceptance_ratio


def run_comparison(num_problems=3):
    """Compare greedy vs MCMC sampling on HumanEval benchmark."""
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        print("Error: Please set XAI_API_KEY environment variable")
        print("Get your API key from: https://console.x.ai/")
        return
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )
    
    # Load HumanEval dataset
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    # Test on first few problems
    for i in range(num_problems):
        problem = dataset[i]
        task_id = problem["task_id"]
        prompt = problem["prompt"]
        
        print(f"\n{'='*80}")
        print(f"Problem {i+1}: {task_id}")
        print('='*80)
        print(f"\nPrompt:\n{prompt[:200]}...")  # Show first 200 chars
        
        # Greedy decoding
        print("\n[GREEDY DECODING]")
        greedy_output = greedy_decode(client, prompt, max_tokens=512)
        print(greedy_output[:300] + "..." if len(greedy_output) > 300 else greedy_output)
        
        # MCMC sampling
        print("\n[MCMC SAMPLING]")
        mcmc_output, _, _, acceptance_ratio = mcmc_sampling(
            client, prompt, max_tokens=512, alpha=4.0, mcmc_steps=10
        )
        print(mcmc_output[:300] + "..." if len(mcmc_output) > 300 else mcmc_output)
        
        # Show if outputs differ
        print(f"\n[COMPARISON]")
        if greedy_output == mcmc_output:
            print("✓ Same output")
        else:
            print("✗ Different outputs")
            print(f"  Greedy length: {len(greedy_output)} chars")
            print(f"  MCMC length: {len(mcmc_output)} chars")


if __name__ == "__main__":
    run_comparison(num_problems=3)
