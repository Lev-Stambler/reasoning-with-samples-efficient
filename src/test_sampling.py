import os
import random
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


def mcmc_sampling(client, prompt, max_tokens=512, temperature=0.8, mcmc_steps=3):
    """
    MCMC-inspired sampling: generate multiple samples and accept/reject.
    Simulates MCMC by generating variants and comparing their likelihoods.
    """
    # Initial sample with temperature
    initial_response = client.chat.completions.create(
        model=GROK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=5,
    )
    
    current_text = initial_response.choices[0].message.content
    current_logprob = sum([
        token.logprob for token in initial_response.choices[0].logprobs.content
    ]) if initial_response.choices[0].logprobs else 0
    
    # MCMC refinement steps
    for step in range(mcmc_steps):
        # Generate alternative sample
        proposal_response = client.chat.completions.create(
            model=GROK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature * 1.2,  # Slightly higher temp for exploration
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5,
        )
        
        proposal_text = proposal_response.choices[0].message.content
        proposal_logprob = sum([
            token.logprob for token in proposal_response.choices[0].logprobs.content
        ]) if proposal_response.choices[0].logprobs else 0
        
        # Metropolis-Hastings acceptance
        log_ratio = proposal_logprob - current_logprob
        accept_prob = min(1.0, 2.718 ** log_ratio)  # exp(log_ratio)
        
        if random.random() < accept_prob:
            current_text = proposal_text
            current_logprob = proposal_logprob
            print(f"  [Step {step+1}] Accepted (log_ratio: {log_ratio:.3f})")
        else:
            print(f"  [Step {step+1}] Rejected (log_ratio: {log_ratio:.3f})")
    
    return current_text


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
        mcmc_output = mcmc_sampling(client, prompt, max_tokens=512, temperature=0.8, mcmc_steps=3)
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
