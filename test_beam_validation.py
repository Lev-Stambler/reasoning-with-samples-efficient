#!/usr/bin/env python3
"""
Test script to verify beam search validation functionality with entropy checking.
"""
import os
from src.benchmark_runner import BeamSearchSampling
from openai import OpenAI

def test_beam_validation():
    """Test beam search with candidate validation and entropy metrics."""
    
    # Setup API client (adjust based on your configuration)
    api_key = os.environ.get("XAI_API_KEY", "dummy-key")
    base_url = os.environ.get("API_BASE_URL", "https://api.x.ai/v1")
    model = os.environ.get("MODEL_NAME", "grok-beta")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    client.default_model = model
    
    # Create beam search with validation enabled
    beam_search = BeamSearchSampling(
        alpha=4.0,
        beam_width=2,
        n_per_beam=2,
        tokens_per_step=50,  # Smaller for testing
        validate_candidates=True,  # Enable validation
        debug=True,  # Show debug output (includes entropy and confidence)
        supports_n_param=True,
    )
    
    # Simple test prompt
    test_prompt = """Write a Python function that calculates the factorial of a number."""
    
    print("=" * 80)
    print("Testing Beam Search with Candidate Validation + Entropy Analysis")
    print("=" * 80)
    print(f"\nPrompt: {test_prompt}")
    print("\nGenerating with validation enabled...")
    print("\nValidation will show:")
    print("  - Response: Yes/No (is candidate on right track)")
    print("  - Entropy: measures uncertainty in model's answer (lower = more certain)")
    print("  - Confidence: probability of chosen token (higher = more confident)")
    print("-" * 80)
    
    # Generate
    completion, prompt_tokens, completion_tokens = beam_search.generate(
        client, 
        test_prompt, 
        max_tokens=200
    )
    
    print("\n" + "=" * 80)
    print("FINAL RESULT:")
    print("=" * 80)
    print(completion)
    print("\n" + "=" * 80)
    print(f"Tokens used: prompt={prompt_tokens}, completion={completion_tokens}, total={prompt_tokens + completion_tokens}")
    print("=" * 80)
    print("\nInterpretation:")
    print("  - Low entropy (< 0.5): Model is very confident in Yes/No answer")
    print("  - High entropy (> 1.5): Model is uncertain about the answer")
    print("  - High confidence (> 0.9): Strong belief in the chosen response")
    print("=" * 80)

if __name__ == "__main__":
    test_beam_validation()
