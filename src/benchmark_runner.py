import os
import time
import json
import tempfile
import asyncio
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset
import random
import re
import numpy as np

from async_openai_client import AsyncOpenAIClient, ChatCompletionResponse


# Pricing per 1M tokens (input, output) in USD
MODEL_PRICING = {
    "grok-beta": (3.00, 15.00),  # High-end reasoning model (likely alias for base grok-4)
    "grok-2-1212": (2.00, 10.00),
    "grok-2-latest": (2.00, 10.00),
    "grok-4-1-fast-non-reasoning": (0.20, 0.50),  # Fast, low-latency variant (10-20x cheaper!)
    "grok-4-1-fast-reasoning": (0.20, 0.50),  # Fast reasoning variant with chain-of-thought
    "gpt-4": (30.00, 60.00),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "claude-3-opus": (15.00, 75.00),
    "claude-3-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    # Default pricing for unknown models
    "default": (2.00, 10.00),
}


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate cost in USD for API usage.
    
    Args:
        model_name: Name of the model
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
    
    Returns:
        Cost in USD
    """
    # Get pricing or use default
    pricing = MODEL_PRICING.get(model_name, MODEL_PRICING["default"])
    input_price_per_million, output_price_per_million = pricing
    
    # Calculate cost
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (completion_tokens / 1_000_000) * output_price_per_million
    
    return input_cost + output_cost


@dataclass
class SamplingResult:
    """Results for a single problem with a specific sampling strategy."""
    task_id: str
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_seconds: float
    cost_usd: float
    passed: bool = False
    metadata: Optional[Dict] = None  # For benchmark-specific data
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "completion": self.completion,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "time_seconds": self.time_seconds,
            "cost_usd": self.cost_usd,
            "passed": self.passed,
            "metadata": self.metadata or {}
        }


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
    total_cost: float
    cost_per_problem: float
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
        restrict_to_last_n: int = None,  # Only resample last N blocks (None = disabled)
        block_size: int = 192,  # Block size B for block-wise generation (paper default)
        debug: bool = False,  # Print debug info during MCMC
    ):
        name = f"MCMC(α={alpha},steps={mcmc_steps},B={block_size})"
        if restrict_to_last_n is not None:
            name += f",lastN={restrict_to_last_n}"
        super().__init__(name)
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.top_logprobs = top_logprobs
        # Support legacy 'temperature' parameter as alias for proposal_temperature
        self.proposal_temperature = temperature if temperature is not None else proposal_temperature
        self.restrict_to_last_n = restrict_to_last_n
        self.block_size = block_size
        self.debug = debug

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
        # Track if completion ended naturally (EOS) vs hitting max_tokens
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            text, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
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
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            continuation, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using MCMC power sampling with block-wise generation.

        Algorithm (matching paper):
        1. Generate tokens block-by-block (B tokens per block)
        2. After each block, run MCMC refinement steps
        3. MCMC uses block-aligned index selection (idx = block_idx * B)
        4. After all blocks, truncate at EOS if present
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        attempts = 0
        acceptances = 0

        # Initialize with empty generation
        tokens_cur = []
        log_p_cur = []
        log_target_cur = []

        # Calculate number of blocks to generate
        num_blocks_to_generate = max_tokens // self.block_size
        if num_blocks_to_generate < 1:
            num_blocks_to_generate = 1

        if self.debug:
            print(f"[MCMC] Block-wise generation: {num_blocks_to_generate} blocks of {self.block_size} tokens")

        # Generate block by block
        for block_num in range(num_blocks_to_generate):
            # Generate next block
            prefix = "".join(tokens_cur) if tokens_cur else ""

            if block_num == 0:
                # First block: use _sample_full (no prefix)
                block_text, block_tokens, block_log_p, block_log_target, pt, ct, _ = self._sample_full(
                    client, prompt, self.block_size
                )
            else:
                pass
                # Subsequent blocks: continue from prefix
                block_text, block_tokens, block_log_p, block_log_target, pt, ct, _ = self._sample_continuation(
                    client, prompt, prefix, self.block_size
                )

            total_prompt_tokens += pt
            total_completion_tokens += ct

            # Extend current state with new block
            tokens_cur.extend(block_tokens)
            log_p_cur.extend(block_log_p)
            log_target_cur.extend(block_log_target)

            if self.debug:
                print(f"[MCMC] Block {block_num+1}/{num_blocks_to_generate}: generated {len(block_tokens)} tokens, total={len(tokens_cur)}")

            # Run MCMC refinement steps on current state
            for step in range(self.mcmc_steps):
                # Block-aligned index selection
                num_complete_blocks = len(tokens_cur) // self.block_size
                attempts += 1

                # Pick random block boundary (keep at least first block)
                # If restrict_to_last_n is set, only resample from last N blocks
                if self.restrict_to_last_n is not None:
                    min_block = max(1, num_complete_blocks - self.restrict_to_last_n)
                else:
                    min_block = 0

                # Check if we have a valid range
                if min_block > num_complete_blocks - 1:
                    if self.debug:
                        print(f"[MCMC]   Step {step+1}: Skipping, restrict_to_last_n={self.restrict_to_last_n} too small")
                    continue

                block_idx = random.randint(min_block, num_complete_blocks - 1)
                idx = block_idx * self.block_size

                # Prefix to keep (as text)
                prefix = "".join(tokens_cur[:idx])

                # Target length for proposal (same as current)
                target_len = len(tokens_cur) - idx #+ self.block_size

                # Generate new suffix
                new_suffix, tokens_prop, log_p_prop, log_target_prop, pt, ct, _ = self._sample_continuation(
                    client, prompt, prefix, target_len
                )
                total_prompt_tokens += pt
                total_completion_tokens += ct

                # Slice current suffix to match proposal length (handles variable-length proposals)
                prop_len = len(tokens_prop)
                log_p_cur_suffix = log_p_cur[idx:idx + prop_len]
                log_target_cur_suffix = log_target_cur[idx:idx + prop_len]

                # MH acceptance ratio for suffixes only
                # log A = log(π(suffix')/π(suffix)) + log(q(suffix)/q(suffix'))
                log_r = (
                    sum(log_target_prop) + sum(log_p_cur_suffix)
                    - sum(log_target_cur_suffix) - sum(log_p_prop)
                )

                # Accept with probability min(1, exp(log_r))
                accepted = np.random.rand() < np.exp(log_r)

                if self.debug:
                    status = "ACCEPT" if accepted else "REJECT"
                    print(f"[MCMC]   Step {step+1}: block_idx={block_idx}, idx={idx}, log_r={log_r:.3f}, {status}")

                if accepted:
                    acceptances += 1
                    # Update current state with new suffix
                    tokens_cur = tokens_cur[:idx] + tokens_prop
                    log_p_cur = log_p_cur[:idx] + log_p_prop
                    log_target_cur = log_target_cur[:idx] + log_target_prop

        # Reconstruct text from final tokens
        current_text = "".join(tokens_cur)

        # Store acceptance ratio for diagnostics
        self._last_acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0

        if self.debug:
            print(f"[MCMC] Final: {len(tokens_cur)} tokens, acceptance={self._last_acceptance_ratio:.1%}")
            print(f"[MCMC] Final text: {current_text[:200]}..." if len(current_text) > 200 else f"[MCMC] Final text: {current_text}")

        return current_text, total_prompt_tokens, total_completion_tokens

    def get_acceptance_ratio(self) -> float:
        """Return the acceptance ratio from the last generate() call."""
        return getattr(self, '_last_acceptance_ratio', 0.0)


@dataclass
class Proposal:
    """A proposal for parallel MCMC."""
    tokens: List[str]
    log_p: List[float]
    log_target: List[float]
    prompt_tokens: int
    completion_tokens: int


class ParallelMCMCSampling(SamplingStrategy):
    """
    Multiple Proposals MCMC with parallel suffix generation.

    Implements the generalized Metropolis-Hastings algorithm from:
    "A general construction for parallelizing Metropolis-Hastings algorithms" (Calderhead 2014, PNAS)

    For each MCMC step:
    1. Generate N-1 new proposals in parallel + keep current state
    2. Compute pairwise acceptance ratios R(i,j) = π(x_j)/π(x_i)
    3. Construct transition matrix A(i,j) = (1/N) * min(1, R(i,j))
    4. Sample next state from row of transition matrix

    Target distribution: π(x) = p(x)^α (power sampling)
    """

    def __init__(
        self,
        alpha: float = 1.67,
        mcmc_steps: int = 5,
        top_logprobs: int = 5,
        proposal_temperature: float = 0.59,
        block_size: int = 192,
        debug: bool = False,
        # Parallel-specific params:
        num_proposals: int = 4,
        max_concurrent: int = 100,
        timeout: float = 60.0,
        max_retries: int = 3,
        length_normalize: bool = False,
        # API config (set by generate() from client):
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
        model: str = "grok-4-1-fast-non-reasoning",
    ):
        name = f"ParallelMCMC(α={alpha},steps={mcmc_steps},B={block_size},N={num_proposals})"
        super().__init__(name)
        self.alpha = alpha
        self.mcmc_steps = mcmc_steps
        self.top_logprobs = top_logprobs
        self.proposal_temperature = proposal_temperature
        self.block_size = block_size
        self.debug = debug
        self.num_proposals = num_proposals
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.length_normalize = length_normalize
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def _extract_logprobs_from_response(self, response: ChatCompletionResponse) -> Tuple[List[str], List[float], List[float]]:
        """Extract tokens and logprobs from async API response."""
        tokens = response.tokens
        log_p = response.log_probs
        log_target = [self.alpha * lp for lp in log_p]
        return tokens, log_p, log_target

    async def _sample_full_async(
        self,
        client: AsyncOpenAIClient,
        prompt: str,
        max_tokens: int,
    ) -> Tuple[str, List[str], List[float], List[float], int, int, bool]:
        """Generate a full sample from base model (async)."""
        response = await client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        tokens, log_p, log_target = self._extract_logprobs_from_response(response)
        finished_naturally = response.finish_reason == "stop"
        return (
            response.text, tokens, log_p, log_target,
            response.prompt_tokens, response.completion_tokens,
            finished_naturally
        )

    async def _sample_continuation_async(
        self,
        client: AsyncOpenAIClient,
        prompt: str,
        prefix: str,
        max_tokens: int,
    ) -> Tuple[str, List[str], List[float], List[float], int, int, bool]:
        """Generate continuation from prefix (async)."""
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix},
            ],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        tokens, log_p, log_target = self._extract_logprobs_from_response(response)
        finished_naturally = response.finish_reason == "stop"
        return (
            response.text, tokens, log_p, log_target,
            response.prompt_tokens, response.completion_tokens,
            finished_naturally
        )

    async def _generate_parallel_proposals(
        self,
        client: AsyncOpenAIClient,
        prompt: str,
        prefix: str,
        target_len: int,
        current_proposal: Proposal,
    ) -> Tuple[List[Proposal], int, int]:
        """
        Generate N-1 new proposals in parallel + include current state.

        Returns:
            (proposals, total_prompt_tokens, total_completion_tokens)
            proposals[0] is always the current state
        """
        # Create N-1 async requests
        requests = []
        for _ in range(self.num_proposals - 1):
            requests.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix},
                ],
                "temperature": self.proposal_temperature,
                "max_tokens": target_len,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
            })

        # Execute in parallel
        responses = await client.chat_completion_batch(requests, max_concurrent=self.max_concurrent)

        # Parse responses into Proposals
        proposals = [current_proposal]  # Index 0 is current state
        total_pt = 0
        total_ct = 0

        for response in responses:
            tokens, log_p, log_target = self._extract_logprobs_from_response(response)
            proposals.append(Proposal(
                tokens=tokens,
                log_p=log_p,
                log_target=log_target,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            ))
            total_pt += response.prompt_tokens
            total_ct += response.completion_tokens

        return proposals, total_pt, total_ct

    async def _generate_parallel_proposals_full(
        self,
        client: AsyncOpenAIClient,
        prompt: str,
        target_len: int,
        current_proposal: Proposal,
    ) -> Tuple[List[Proposal], int, int]:
        """
        Generate N-1 new full proposals in parallel (no prefix) + include current state.

        Used when regenerating from the beginning (idx=0).
        """
        # Create N-1 async requests for full generation
        requests = []
        for _ in range(self.num_proposals - 1):
            requests.append({
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.proposal_temperature,
                "max_tokens": target_len,
                "logprobs": True,
                "top_logprobs": self.top_logprobs,
            })

        # Execute in parallel
        responses = await client.chat_completion_batch(requests, max_concurrent=self.max_concurrent)

        # Parse responses into Proposals
        proposals = [current_proposal]  # Index 0 is current state
        total_pt = 0
        total_ct = 0

        for response in responses:
            tokens, log_p, log_target = self._extract_logprobs_from_response(response)
            proposals.append(Proposal(
                tokens=tokens,
                log_p=log_p,
                log_target=log_target,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            ))
            total_pt += response.prompt_tokens
            total_ct += response.completion_tokens

        return proposals, total_pt, total_ct

    def _compute_transition_matrix(self, proposals: List[Proposal]) -> np.ndarray:
        """
        Compute transition matrix A(i,j) for multiple proposals.

        A(i,j) = (1/N) * min(1, R(i,j))  for j ≠ i
        A(i,i) = 1 - Σ_{j≠i} A(i,j)

        For power sampling with symmetric proposal:
        R(i,j) = π(x_j)/π(x_i) = exp(α * (sum(log_p_j) - sum(log_p_i)))

        Note: We use log_target which already has α applied, so:
        log R(i,j) = sum(log_target_j) - sum(log_target_i)

        If length_normalize=True, we normalize by length to remove length penalty:
        log R(i,j) = mean(log_target_j) - mean(log_target_i)
        """
        N = len(proposals)
        A = np.zeros((N, N))

        # Compute log target sums (or means if length_normalize) for each proposal
        if self.length_normalize:
            log_target_sums = np.array([
                sum(p.log_target) / len(p.log_target) if p.log_target else 0.0
                for p in proposals
            ])
        else:
            log_target_sums = np.array([sum(p.log_target) for p in proposals])

        for i in range(N):
            for j in range(N):
                if i != j:
                    # log R(i,j) = log_target_j - log_target_i
                    log_R = log_target_sums[j] - log_target_sums[i]
                    # Clamp for numerical stability (avoid overflow)
                    log_R = np.clip(log_R, -700, 700)
                    # A(i,j) = (1/N) * min(1, exp(log_R))
                    A[i, j] = (1.0 / N) * min(1.0, np.exp(log_R))

            # A(i,i) = 1 - sum of other transitions
            A[i, i] = 1.0 - np.sum(A[i, :])

        # Ensure non-negative (numerical precision)
        A = np.maximum(A, 0.0)

        return A

    def _sample_next_state(self, transition_matrix: np.ndarray, current_idx: int) -> int:
        """
        Sample next state from transition matrix row.

        Args:
            transition_matrix: N x N transition matrix
            current_idx: Index of current state

        Returns:
            Index of next state
        """
        probs = transition_matrix[current_idx, :]
        # Ensure probabilities sum to 1 (handle numerical errors)
        probs = probs / probs.sum()
        return np.random.choice(len(probs), p=probs)

    async def _generate_async(
        self,
        api_key: str,
        base_url: str,
        model: str,
        prompt: str,
        max_tokens: int,
    ) -> Tuple[str, int, int]:
        """
        Async implementation of block-wise generation with parallel MCMC.
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0
        attempts = 0
        acceptances = 0

        async with AsyncOpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=self.timeout,
            max_retries=self.max_retries,
        ) as client:
            # Initialize with empty generation
            tokens_cur: List[str] = []
            log_p_cur: List[float] = []
            log_target_cur: List[float] = []

            # Calculate number of blocks to generate
            num_blocks_to_generate = max_tokens // self.block_size
            if num_blocks_to_generate < 1:
                num_blocks_to_generate = 1

            if self.debug:
                print(f"[ParallelMCMC] Block-wise generation: {num_blocks_to_generate} blocks of {self.block_size} tokens, N={self.num_proposals}")

            # Generate block by block
            for block_num in range(num_blocks_to_generate):
                prefix = "".join(tokens_cur) if tokens_cur else ""

                if block_num == 0:
                    # First block: use _sample_full_async
                    _, block_tokens, block_log_p, block_log_target, pt, ct, _ = await self._sample_full_async(
                        client, prompt, self.block_size
                    )
                else:
                    # Subsequent blocks: continue from prefix
                    _, block_tokens, block_log_p, block_log_target, pt, ct, _ = await self._sample_continuation_async(
                        client, prompt, prefix, self.block_size
                    )

                total_prompt_tokens += pt
                total_completion_tokens += ct

                # Extend current state with new block
                tokens_cur.extend(block_tokens)
                log_p_cur.extend(block_log_p)
                log_target_cur.extend(block_log_target)

                if self.debug:
                    print(f"[ParallelMCMC] Block {block_num+1}/{num_blocks_to_generate}: generated {len(block_tokens)} tokens, total={len(tokens_cur)}")

                # Run MCMC refinement steps with parallel proposals
                for step in range(self.mcmc_steps):
                    num_complete_blocks = len(tokens_cur) // self.block_size

                    # Need at least 1 complete block to do MCMC
                    # (model may finish early with fewer tokens than block_size)
                    if num_complete_blocks < 1:
                        if self.debug:
                            print(f"[ParallelMCMC]   Step {step+1}: Skipping MCMC, {len(tokens_cur)} tokens < block_size {self.block_size}")
                        break  # No point continuing MCMC steps for this block

                    attempts += 1

                    # Pick random block boundary (can regenerate from beginning with min_block=0)
                    min_block = 0
                    block_idx = random.randint(min_block, num_complete_blocks - 1)
                    idx = block_idx * self.block_size

                    # Prefix to keep
                    prefix = "".join(tokens_cur[:idx])
                    # Target length includes an extra block for exploration
                    target_len = len(tokens_cur) - idx #+ self.block_size

                    # Current suffix as Proposal (will be sliced to match proposal length later)
                    current_proposal = Proposal(
                        tokens=tokens_cur[idx:],
                        log_p=log_p_cur[idx:],
                        log_target=log_target_cur[idx:],
                        prompt_tokens=0,
                        completion_tokens=0,
                    )

                    # Generate N-1 new proposals in parallel
                    # If idx=0 (regenerating from beginning), use full generation instead of continuation
                    if idx == 0:
                        proposals, pt, ct = await self._generate_parallel_proposals_full(
                            client, prompt, target_len, current_proposal
                        )
                    else:
                        proposals, pt, ct = await self._generate_parallel_proposals(
                            client, prompt, prefix, target_len, current_proposal
                        )
                    total_prompt_tokens += pt
                    total_completion_tokens += ct

                    # For fair comparison, slice current suffix to match minimum proposal length
                    # (handles variable-length proposals)
                    min_len = min(len(p.tokens) for p in proposals)
                    for i, p in enumerate(proposals):
                        if len(p.tokens) > min_len:
                            proposals[i] = Proposal(
                                tokens=p.tokens[:min_len],
                                log_p=p.log_p[:min_len],
                                log_target=p.log_target[:min_len],
                                prompt_tokens=p.prompt_tokens,
                                completion_tokens=p.completion_tokens,
                            )

                    # Compute transition matrix
                    A = self._compute_transition_matrix(proposals)

                    # Sample next state (current is at index 0)
                    next_idx = self._sample_next_state(A, current_idx=0)
                    accepted = (next_idx != 0)

                    if self.debug:
                        log_targets = [sum(p.log_target) for p in proposals]
                        status = f"ACCEPT({next_idx})" if accepted else "STAY"
                        print(f"[ParallelMCMC]   Step {step+1}: block_idx={block_idx}, log_targets={[f'{lt:.1f}' for lt in log_targets]}, {status}")

                    if accepted:
                        acceptances += 1
                        selected = proposals[next_idx]
                        tokens_cur = tokens_cur[:idx] + selected.tokens
                        log_p_cur = log_p_cur[:idx] + selected.log_p
                        log_target_cur = log_target_cur[:idx] + selected.log_target

        # Reconstruct text from final tokens
        current_text = "".join(tokens_cur)

        # Store acceptance ratio
        self._last_acceptance_ratio = acceptances / attempts if attempts > 0 else 0.0

        if self.debug:
            print(f"[ParallelMCMC] Final: {len(tokens_cur)} tokens, acceptance={self._last_acceptance_ratio:.1%}")

        return current_text, total_prompt_tokens, total_completion_tokens

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> Tuple[str, int, int]:
        """
        Generate completion using parallel MCMC with multiple proposals.

        Uses asyncio.run() to bridge sync/async.
        """
        # Extract API config from sync client
        api_key = self.api_key or client.api_key
        base_url = self.base_url or str(client.base_url).rstrip('/')
        model = self.model or getattr(client, 'default_model', 'grok-4-1-fast-non-reasoning')

        # Run async generation
        return asyncio.run(self._generate_async(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        ))

    def get_acceptance_ratio(self) -> float:
        """Return the acceptance ratio from the last generate() call."""
        return getattr(self, '_last_acceptance_ratio', 0.0)


class BeamSearchSampling(SamplingStrategy):
    """
    Beam search with power sampling via API.

    Maintains beam_width parallel hypotheses, scores using p^α logprobs,
    and uses length normalization to prevent short-sequence bias.

    Unlike MCMC which uses accept/reject, beam search deterministically
    keeps the top-k hypotheses at each step.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        beam_width: int = 2,
        n_per_beam: int = 2,  # Generate n continuations per beam
        tokens_per_step: int = 192,  # Generate this many tokens per expansion
        use_length_penalty: bool = True,  # Whether to apply length normalization
        length_penalty: float = 0.6,
        proposal_temperature: float = 1.0,
        top_logprobs: int = 5,
        debug: bool = False,
    ):
        name = f"BeamSearch(α={alpha},width={beam_width},n={n_per_beam},tps={tokens_per_step})"
        super().__init__(name)
        self.alpha = alpha
        self.beam_width = beam_width
        self.n_per_beam = n_per_beam
        self.tokens_per_step = tokens_per_step
        self.use_length_penalty = use_length_penalty
        self.length_penalty = length_penalty
        self.proposal_temperature = proposal_temperature
        self.top_logprobs = top_logprobs
        self.debug = debug

    def _extract_logprobs_with_tokens(self, response) -> tuple[list[str], list[float], list[float]]:
        """Extract tokens and logprobs from API response."""
        if not response.choices[0].logprobs or not response.choices[0].logprobs.content:
            return [], [], []

        tokens = [token.token for token in response.choices[0].logprobs.content]
        log_p = [token.logprob for token in response.choices[0].logprobs.content]
        log_target = [self.alpha * lp for lp in log_p]
        print("AAA", log_p, log_target)

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
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            text, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def _sample_continuation(self, client: OpenAI, prompt: str, prefix: str, max_tokens: int):
        """Generate a single continuation from a prefix."""
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}
            ],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )

        continuation = response.choices[0].message.content
        tokens, log_p, log_target = self._extract_logprobs_with_tokens(response)
        finished_naturally = response.choices[0].finish_reason == "stop"

        return (
            continuation, tokens, log_p, log_target,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            finished_naturally
        )

    def _sample_continuation_multiple(self, client: OpenAI, prompt: str, prefix: str, max_tokens: int, n: int):
        """Generate n continuations from a prefix for true beam search expansion."""
        response = client.chat.completions.create(
            model=client.default_model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}
            ],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            n=n,  # Generate n different samples
        )

        # Extract all n continuations
        results = []
        for choice in response.choices:
            continuation = choice.message.content

            # Extract logprobs for this choice
            if choice.logprobs and choice.logprobs.content:
                tokens = [t.token for t in choice.logprobs.content]
                log_p = [t.logprob for t in choice.logprobs.content]
                log_target = [self.alpha * lp for lp in log_p]
            else:
                tokens = []
                log_p = []
                log_target = []

            finished_naturally = choice.finish_reason == "stop"

            results.append((continuation, tokens, log_p, log_target, finished_naturally))

        # Token usage is for ALL n samples combined
        return results, response.usage.prompt_tokens, response.usage.completion_tokens

    async def _sample_continuation_multiple_async(self, client: AsyncOpenAI, prompt: str, prefix: str, max_tokens: int, n: int):
        """Async version: Generate n continuations from a prefix for true beam search expansion."""
        response = await client.chat.completions.create(
            model=client.default_model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}
            ],
            temperature=self.proposal_temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=self.top_logprobs,
            n=n,  # Generate n different samples
        )

        # Extract all n continuations (same logic as sync version)
        results = []
        for choice in response.choices:
            continuation = choice.message.content

            # Extract logprobs for this choice
            if choice.logprobs and choice.logprobs.content:
                tokens = [t.token for t in choice.logprobs.content]
                log_p = [t.logprob for t in choice.logprobs.content]
                log_target = [self.alpha * lp for lp in log_p]
            else:
                tokens = []
                log_p = []
                log_target = []

            finished_naturally = choice.finish_reason == "stop"

            results.append((continuation, tokens, log_p, log_target, finished_naturally))

        # Token usage is for ALL n samples combined
        return results, response.usage.prompt_tokens, response.usage.completion_tokens

    async def _expand_beams_parallel(self, client: AsyncOpenAI, active_beams, prompt, block_num):
        """Parallelize beam expansion using async/await."""
        tasks = []

        for beam_text, beam_tokens, beam_log_p, beam_log_target, _ in active_beams:
            if block_num == 0 and not beam_text:
                # First expansion: generate from scratch with n samples
                task = self._sample_continuation_multiple_async(
                    client, prompt, "", self.tokens_per_step, n=self.n_per_beam
                )
            else:
                # Subsequent expansions: generate n continuations from this beam
                task = self._sample_continuation_multiple_async(
                    client, prompt, beam_text, self.tokens_per_step, n=self.n_per_beam
                )
            tasks.append((task, (beam_text, beam_tokens, beam_log_p, beam_log_target)))

        # Run all API calls in parallel!
        results = await asyncio.gather(*[task for task, _ in tasks])

        # Process results and create candidate beams
        candidate_beams = []
        total_pt = 0
        total_ct = 0

        for i, (continuations, pt, ct) in enumerate(results):
            total_pt += pt
            total_ct += ct
            beam_text, beam_tokens, beam_log_p, beam_log_target = tasks[i][1]

            for text, tokens, log_p, log_target, finished in continuations:
                if block_num == 0:
                    # First block
                    candidate_beams.append((text, tokens, log_p, log_target, finished))
                else:
                    # Subsequent blocks - concatenate with beam prefix
                    new_text = beam_text + text
                    new_tokens = beam_tokens + tokens
                    new_log_p = beam_log_p + log_p
                    new_log_target = beam_log_target + log_target
                    candidate_beams.append((new_text, new_tokens, new_log_p, new_log_target, finished))

        return candidate_beams, total_pt, total_ct

    def _calculate_beam_score(self, log_target: list[float], length: int) -> float:
        """Calculate beam score, optionally with length normalization."""
        if length == 0:
            return float('-inf')  # Empty beams should never be selected
        cumulative_score = sum(log_target)
        if self.use_length_penalty:
            normalized_score = cumulative_score / (length ** self.length_penalty)
            return normalized_score
        else:
            return cumulative_score

    async def _generate_async(self, client: AsyncOpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Async version of generate() with parallel beam expansion.

        Algorithm:
        1. Start with empty beam
        2. For each beam, generate n_per_beam continuations IN PARALLEL (beam branching)
        3. Score all beam_width × n_per_beam candidates using p^α
        4. Keep top beam_width beams by score
        5. Repeat until max_tokens or all beams finish

        With beam_width=2, n_per_beam=2: generates 4 candidates per iteration in parallel.
        """
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Initialize beams: (text, tokens, log_p, log_target, finished)
        active_beams = [("", [], [], [], False)]
        completed_beams = []

        num_expansions = 0
        num_blocks = max_tokens // self.tokens_per_step
        if num_blocks < 1:
            num_blocks = 1

        if self.debug:
            print(f"[BeamSearch] TRUE beam search (ASYNC): beam_width={self.beam_width}, n_per_beam={self.n_per_beam}")
            print(f"[BeamSearch] Generating {num_blocks} blocks of {self.tokens_per_step} tokens")
            print(f"[BeamSearch] α={self.alpha}, length_penalty={self.length_penalty}")
            print(f"[BeamSearch] Each iteration: {len(active_beams)} beams × {self.n_per_beam} samples = candidates (PARALLEL)")

        for block_num in range(num_blocks):
            if not active_beams:
                break

            # PARALLEL EXPANSION PHASE: Generate n_per_beam continuations for ALL beams at once
            candidate_beams, pt, ct = await self._expand_beams_parallel(
                client, active_beams, prompt, block_num
            )
            total_prompt_tokens += pt
            total_completion_tokens += ct

            num_expansions += 1

            if self.debug:
                print(f"[BeamSearch] Block {block_num+1}: Generated {len(candidate_beams)} candidates (parallel)")

            # Score all candidates
            scored_beams = []
            for beam in candidate_beams:
                text, tokens, log_p, log_target, finished = beam
                score = self._calculate_beam_score(log_target, len(tokens))
                scored_beams.append((score, text, tokens, log_p, log_target, finished))

            # Sort by score (descending)
            scored_beams.sort(key=lambda x: x[0], reverse=True)

            # Separate completed and active
            new_completed = [b for b in scored_beams if b[5]]
            new_active = [b for b in scored_beams if not b[5]]

            # Keep top beams
            completed_beams.extend(new_completed[:self.beam_width])
            active_beams = [(b[1], b[2], b[3], b[4], b[5]) for b in new_active[:self.beam_width]]

            if self.debug:
                print(f"[BeamSearch] Block {block_num+1}/{num_blocks}: "
                      f"{len(active_beams)} active, {len(completed_beams)} completed")
                if scored_beams:
                    best_score = scored_beams[0][0]
                    print(f"[BeamSearch]   Best score: {best_score:.4f}")

            # Stop if we have enough completed beams
            if len(completed_beams) >= self.beam_width:
                if self.debug:
                    print(f"[BeamSearch] Stopping: {len(completed_beams)} beams completed")
                break

        # Select best beam from all (completed + active)
        all_beams = completed_beams + [
            (self._calculate_beam_score(beam[3], len(beam[1])),) + beam
            for beam in active_beams
        ]

        if not all_beams:
            # Fallback: return empty
            return "", total_prompt_tokens, total_completion_tokens

        # Sort by score and take best
        all_beams.sort(key=lambda x: x[0], reverse=True)
        best_beam = all_beams[0]
        best_text = best_beam[1]

        # Store metadata for diagnostics
        self._last_num_expansions = num_expansions
        self._last_best_score = best_beam[0]
        self._last_num_completed = len(completed_beams)

        if self.debug:
            print(f"[BeamSearch] Final: {len(best_beam[2])} tokens, score={best_beam[0]:.4f}")
            print(f"[BeamSearch] Text: {best_text[:200]}..." if len(best_text) > 200 else f"[BeamSearch] Text: {best_text}")

        return best_text, total_prompt_tokens, total_completion_tokens

    async def _run_with_client(self, api_key: str, base_url: str, model: str, prompt: str, max_tokens: int):
        """Helper to run async generation with proper client lifecycle."""
        async with AsyncOpenAI(api_key=api_key, base_url=base_url) as async_client:
            async_client.default_model = model
            return await self._generate_async(async_client, prompt, max_tokens)

    def generate(self, client: OpenAI, prompt: str, max_tokens: int = 512) -> tuple[str, int, int]:
        """
        Generate completion using TRUE beam search with power sampling (with async parallelization).

        This method wraps the async implementation and creates an AsyncOpenAI client automatically.

        Algorithm:
        1. Start with empty beam
        2. For each beam, generate n_per_beam continuations IN PARALLEL (beam branching)
        3. Score all beam_width × n_per_beam candidates using p^α
        4. Keep top beam_width beams by score
        5. Repeat until max_tokens or all beams finish

        With beam_width=2, n_per_beam=2: generates 4 candidates per iteration in parallel.
        """
        # Run async version with proper client lifecycle management
        return asyncio.run(
            self._run_with_client(
                api_key=client.api_key,
                base_url=str(client.base_url) if client.base_url else "https://api.openai.com/v1",
                model=client.default_model,
                prompt=prompt,
                max_tokens=max_tokens
            )
        )

    def get_num_expansions(self) -> int:
        """Return the number of expansions from the last generate() call."""
        return getattr(self, '_last_num_expansions', 0)

    def get_best_score(self) -> float:
        """Return the best beam score from the last generate() call."""
        return getattr(self, '_last_best_score', 0.0)


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
    
    @abstractmethod
    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """
        Format a prediction for official evaluation tools.
        Returns: Dictionary in the format expected by the benchmark's evaluator.
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
        """
        DEPRECATED: Use official evaluation instead.
        This method is not reliable - use format_prediction() and official evaluators.
        """
        # Return None to indicate evaluation should be done externally
        return False, "use_official_evaluator"
    
    def format_prediction(self, problem: Dict, completion: str) -> Dict:
        """Format prediction for HumanEval official evaluator."""
        return {
            "task_id": problem["task_id"],
            "completion": completion
        }


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
        base_url: str = "https://api.x.ai/v1",
        output_dir: str = "predictions",
        prompt_prefix: str = "",
        prompt_suffix: str = ""
    ):
        self.benchmark = benchmark
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.client.default_model = model_name
        self.results: List[SamplingResult] = []
        self.output_dir = output_dir
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Apply custom prefix/suffix if provided
        if self.prompt_prefix:
            prompt = self.prompt_prefix + prompt
        if self.prompt_suffix:
            prompt = prompt + self.prompt_suffix
        
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
        
        # Calculate cost
        cost = calculate_cost(self.model_name, prompt_tokens, completion_tokens)
        
        return SamplingResult(
            task_id=task_id,
            completion=extracted_completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            time_seconds=elapsed_time,
            cost_usd=cost,
            passed=passed,
            metadata={"result_message": result_msg}
        )
    
    def run_benchmark(
        self,
        strategies: List[SamplingStrategy],
        num_problems: int = 10,
        max_tokens: int = 512,
        run_id: str = None
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Run benchmark for multiple strategies.
        Generates prediction files for official evaluation.
        Returns: Dict mapping strategy name to metrics.
        """
        # Load benchmark dataset
        print(f"Loading {self.benchmark.name()} dataset...")
        self.benchmark.load_dataset()
        
        results_by_strategy: Dict[str, List[SamplingResult]] = {s.name: [] for s in strategies}
        predictions_by_strategy: Dict[str, List[Dict]] = {s.name: [] for s in strategies}
        
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
                    print(f"{status} ({result.time_seconds:.2f}s, {result.total_tokens} tokens, ${result.cost_usd:.4f})")
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
            
            total_cost = sum(r.cost_usd for r in results)
            
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
                total_cost=total_cost,
                cost_per_problem=total_cost / len(results),
                num_problems=len(results)
            )
        
        return metrics
    
    def save_predictions(self, predictions: List[Dict], strategy_name: str, run_id: str = None):
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
