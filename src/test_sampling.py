import torch
import torch.nn.functional as F
from llm_wrapper import CustomSamplerLLM


def greedy_decode(llm, prompt, max_tokens=50):
    """Greedy decoding: always pick the most likely token."""
    input_ids = llm.tokenizer.encode(prompt, return_tensors="pt").to(llm.device)
    
    for _ in range(max_tokens):
        outputs = llm.model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == llm.tokenizer.eos_token_id:
            break
    
    return llm.tokenizer.decode(input_ids[0], skip_special_tokens=True)


def mcmc_sampling(llm, prompt, max_tokens=50, temperature=0.8, mcmc_steps=3):
    """
    Simple MCMC-inspired sampling: generate tokens, then refine via accept/reject.
    """
    input_ids = llm.tokenizer.encode(prompt, return_tensors="pt").to(llm.device)
    prompt_len = input_ids.size(1)
    
    # Initial generation with sampling
    for _ in range(max_tokens):
        outputs = llm.model(input_ids)
        next_token_logits = outputs.logits[:, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == llm.tokenizer.eos_token_id:
            break
    
    # MCMC refinement: randomly resample from a position
    for _ in range(mcmc_steps):
        if input_ids.size(1) <= prompt_len + 1:
            break
        
        # Pick a random position to resample from
        resample_pos = torch.randint(prompt_len, input_ids.size(1) - 1, (1,)).item()
        prefix = input_ids[:, :resample_pos]
        
        # Get log prob of current continuation
        outputs_current = llm.model(input_ids[:, :resample_pos + 1])
        logits_current = outputs_current.logits[:, -1, :] / temperature
        current_token = input_ids[:, resample_pos]
        log_prob_current = F.log_softmax(logits_current, dim=-1)[0, current_token]
        
        # Propose new token
        probs_new = F.softmax(logits_current, dim=-1)
        new_token = torch.multinomial(probs_new, num_samples=1)
        log_prob_new = F.log_softmax(logits_current, dim=-1)[0, new_token]
        
        # Accept/reject with Metropolis-Hastings ratio
        log_ratio = log_prob_new - log_prob_current
        if torch.rand(1).item() < torch.exp(log_ratio).item():
            # Accept: rebuild sequence with new token
            input_ids = torch.cat([prefix, new_token], dim=1)
    
    return llm.tokenizer.decode(input_ids[0], skip_special_tokens=True)


def run_comparison(model_name="microsoft/Phi-3-mini-4k-instruct"):
    """Compare greedy vs MCMC sampling on a simple prompt."""
    print(f"Loading model: {model_name}...")
    llm = CustomSamplerLLM(model_name)
    
    prompts = [
        "The capital of France is",
        "To solve this problem, first we need to",
        "Once upon a time in a distant galaxy,"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"Test {i}: {prompt}")
        print('='*70)
        
        # Greedy decoding
        print("\n[GREEDY DECODING]")
        greedy_output = greedy_decode(llm, prompt, max_tokens=40)
        print(greedy_output)
        
        # MCMC sampling
        print("\n[MCMC SAMPLING]")
        mcmc_output = mcmc_sampling(llm, prompt, max_tokens=40, temperature=0.8, mcmc_steps=5)
        print(mcmc_output)
        
        # Show difference
        print(f"\n[COMPARISON]")
        print(f"Same output: {greedy_output == mcmc_output}")


if __name__ == "__main__":
    run_comparison()
