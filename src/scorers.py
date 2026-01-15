"""
Scorer abstraction for sampling strategies.

This module provides pluggable scoring strategies that can be used with
BeamSearchSampling, MCMCSampling, and ParallelMCMCSampling.

Scorers compute scores for generated blocks and track cumulative scores
across blocks with optional EMA weighting.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import asyncio


@dataclass
class ScoreResult:
    """Result of scoring a sequence/block."""
    score: float                    # The block score value
    cumulative_score: float         # Accumulated score across all blocks
    metadata: Optional[dict] = None # Optional scorer-specific data


class Scorer(ABC):
    """
    Abstract base class for all scoring strategies.

    Scorers compute scores for generated text blocks and track
    cumulative scores across blocks with optional EMA weighting.
    """

    def __init__(
        self,
        use_length_penalty: bool = False,
        length_penalty: float = 0.6,
        use_ema: bool = False,
        ema_decay: float = 0.9,
    ):
        """
        Initialize scorer.

        Args:
            use_length_penalty: Whether to apply length normalization
            length_penalty: Exponent for length normalization (0.6 = Google NMT default)
            use_ema: Whether to use exponential moving average for cumulative scores
            ema_decay: Decay factor for EMA (0.9 = 10% weight to new, 90% to history)
        """
        self.use_length_penalty = use_length_penalty
        self.length_penalty = length_penalty
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self._cumulative_scores: dict[int, float] = {}

    @abstractmethod
    async def score_block(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        block_text: str,
        block_tokens: List[str],
        block_log_p: List[float],
        sequence_id: int = 0,
    ) -> ScoreResult:
        """
        Score a single block of generated text.

        Args:
            client: API client for any necessary calls
            prompt: Original prompt
            prefix: Text before this block
            block_text: The text of the block to score
            block_tokens: Tokens in the block
            block_log_p: Log probabilities for each token
            sequence_id: Unique ID to track cumulative scores

        Returns:
            ScoreResult with block score and cumulative score
        """
        pass

    def _apply_length_penalty(self, score: float, length: int) -> float:
        """Apply length normalization if enabled."""
        if self.use_length_penalty and length > 0:
            return score / (length ** self.length_penalty)
        return score

    def _update_cumulative(self, sequence_id: int, block_score: float) -> float:
        """
        Update and return cumulative score with optional EMA.

        Standard: cumulative = sum(block_scores)
        EMA: cumulative_t = decay * cumulative_{t-1} + block_score_t

        EMA gives more weight to recent blocks, useful when:
        - Early reasoning may be exploratory/wrong
        - Final steps matter more for correctness
        - Want to recover from early mistakes
        """
        if sequence_id not in self._cumulative_scores:
            self._cumulative_scores[sequence_id] = 0.0

        if self.use_ema:
            self._cumulative_scores[sequence_id] = (
                self.ema_decay * self._cumulative_scores[sequence_id] + block_score
            )
        else:
            self._cumulative_scores[sequence_id] += block_score

        return self._cumulative_scores[sequence_id]

    def reset_sequence(self, sequence_id: int) -> None:
        """Reset cumulative tracking for a sequence."""
        if sequence_id in self._cumulative_scores:
            del self._cumulative_scores[sequence_id]

    def reset_all(self) -> None:
        """Reset all cumulative tracking."""
        self._cumulative_scores.clear()

    def get_cumulative(self, sequence_id: int) -> float:
        """Get current cumulative score for a sequence."""
        return self._cumulative_scores.get(sequence_id, 0.0)

    def get_name(self) -> str:
        """Return a descriptive name for this scorer."""
        return self.__class__.__name__


class PowerScorer(Scorer):
    """
    Existing approach: score = alpha * sum(log_p) across tokens.

    This is the pi(x) = p(x)^alpha formulation from the paper.
    Higher alpha = sharper distribution, more preference for high-probability sequences.

    Length penalty is applied globally to the cumulative score, not per-block.
    """

    def __init__(
        self,
        alpha: float = 4.0,
        use_length_penalty: bool = False,
        length_penalty: float = 0.6,
        use_ema: bool = False,
        ema_decay: float = 0.9,
    ):
        """
        Initialize PowerScorer.

        Args:
            alpha: Power factor for target distribution pi(x) = p(x)^alpha
            use_length_penalty: Whether to apply length normalization
            length_penalty: Exponent for length normalization
            use_ema: Whether to use EMA for cumulative scores
            ema_decay: Decay factor for EMA
        """
        super().__init__(use_length_penalty, length_penalty, use_ema, ema_decay)
        self.alpha = alpha
        # Track raw cumulative values for correct length penalty
        self._sequence_stats: dict[int, tuple[float, int]] = {}  # sequence_id -> (total_log_target, total_tokens)

    async def score_block(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        block_text: str,
        block_tokens: List[str],
        block_log_p: List[float],
        sequence_id: int = 0,
    ) -> ScoreResult:
        """Score using alpha * sum(log_p) with global length penalty."""
        if not block_log_p:
            return ScoreResult(
                score=float('-inf'),
                cumulative_score=self.get_cumulative(sequence_id),
                metadata={"alpha": self.alpha, "num_tokens": 0}
            )

        # Compute block score: alpha * sum(log_p)
        block_log_target = self.alpha * sum(block_log_p)
        block_len = len(block_tokens)

        # Update cumulative stats (raw values)
        current_stats = self._sequence_stats.get(sequence_id, (0.0, 0))
        new_total_log_target = current_stats[0] + block_log_target
        new_total_len = current_stats[1] + block_len
        self._sequence_stats[sequence_id] = (new_total_log_target, new_total_len)

        # Apply length penalty to TOTAL cumulative score (not per-block)
        if self.use_length_penalty and new_total_len > 0:
            cumulative_score = new_total_log_target / (new_total_len ** self.length_penalty)
        else:
            cumulative_score = new_total_log_target

        # For EMA, apply to the cumulative score
        if self.use_ema:
            prev_cumulative = self._cumulative_scores.get(sequence_id, 0.0)
            cumulative_score = self.ema_decay * prev_cumulative + (1 - self.ema_decay) * cumulative_score

        self._cumulative_scores[sequence_id] = cumulative_score

        return ScoreResult(
            score=block_log_target,  # Raw block score (for MH ratio in MCMC)
            cumulative_score=cumulative_score,  # Length-normalized cumulative
            metadata={
                "alpha": self.alpha,
                "num_tokens": block_len,
                "total_tokens": new_total_len,
                "raw_cumulative": new_total_log_target,
            }
        )

    def reset_sequence(self, sequence_id: int) -> None:
        """Reset cumulative tracking for a sequence."""
        super().reset_sequence(sequence_id)
        if sequence_id in self._sequence_stats:
            del self._sequence_stats[sequence_id]

    def reset_all(self) -> None:
        """Reset all cumulative tracking."""
        super().reset_all()
        self._sequence_stats.clear()

    def get_name(self) -> str:
        return f"PowerScorer(alpha={self.alpha})"


class SelfEvalScorer(Scorer):
    """
    New approach: Append a self-evaluation prompt and use P("yes") as score.

    After generating a block, appends a prompt like:
    "Are we on the right track so far? Answer Yes or No:"
    Then uses the log probability of "Yes" (or other positive token) as the score.
    """

    DEFAULT_PROMPTS = {
        "binary": "Is this reasoning correct so far? Answer Yes or No:",
        "confidence": "Rate confidence in this reasoning (1-10):",
        "progress": "Are we making progress toward the solution? Yes or No:",
        "quality": "Is this a high-quality response so far? Yes or No:",
        "track": "Are we on the right track? Yes or No:",
    }

    def __init__(
        self,
        eval_prompt: Optional[str] = None,
        positive_tokens: Optional[List[str]] = None,
        temperature: float = 0.0,
        use_length_penalty: bool = False,
        length_penalty: float = 0.6,
        use_ema: bool = False,
        ema_decay: float = 0.9,
        fallback_score: float = -10.0,
        model: Optional[str] = None,
    ):
        """
        Initialize SelfEvalScorer.

        Args:
            eval_prompt: The evaluation prompt to append. Defaults to binary yes/no.
            positive_tokens: Tokens indicating positive evaluation (e.g., ["Yes", "yes"])
            temperature: Temperature for eval API call (0 = deterministic)
            use_length_penalty: Whether to apply length normalization
            length_penalty: Exponent for length normalization
            use_ema: Whether to use EMA for cumulative scores
            ema_decay: Decay factor for EMA
            fallback_score: Score to use if positive token not found in logprobs
            model: Optional model name override for eval calls
        """
        super().__init__(use_length_penalty, length_penalty, use_ema, ema_decay)
        self.eval_prompt = eval_prompt or self.DEFAULT_PROMPTS["track"]
        self.positive_tokens = positive_tokens or ["Yes", "yes", " Yes", " yes", "YES"]
        self.temperature = temperature
        self.fallback_score = fallback_score
        self.model = model

    async def score_block(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        block_text: str,
        block_tokens: List[str],
        block_log_p: List[float],
        sequence_id: int = 0,
    ) -> ScoreResult:
        """
        Score by appending eval prompt and getting P(positive_token).

        Makes an additional API call to get logprobs for the evaluation.
        """
        # Construct the evaluation context
        full_context = f"{prefix}{block_text}\n\n{self.eval_prompt}"

        # Make API call to get logprobs for next token
        eval_response = await self._get_eval_logprobs(client, prompt, full_context)

        # Extract P(positive_token) from logprobs
        raw_score = self._extract_positive_prob(eval_response)

        # Apply length penalty if needed
        block_score = self._apply_length_penalty(raw_score, len(block_tokens))

        # Update cumulative
        cumulative = self._update_cumulative(sequence_id, block_score)

        return ScoreResult(
            score=block_score,
            cumulative_score=cumulative,
            metadata={
                "eval_prompt": self.eval_prompt,
                "raw_score": raw_score,
                "top_logprobs": eval_response.get("top_logprobs", {}),
                "generated_token": eval_response.get("token", ""),
            }
        )

    async def _get_eval_logprobs(self, client: Any, prompt: str, context: str) -> dict:
        """
        Make API call to get logprobs for evaluation.

        Uses completions API with max_tokens=1 and logprobs to get
        probabilities for the first token after the eval prompt.
        """
        # Get model name
        model = self.model
        if model is None:
            if hasattr(client, 'default_model'):
                model = client.default_model
            elif hasattr(client, 'model'):
                model = client.model
            else:
                model = "default"

        # Try to apply chat template if available
        if hasattr(client, '_apply_chat_template'):
            full_prompt = client._apply_chat_template(prompt, context)
        else:
            full_prompt = f"{prompt}\n\n{context}"

        try:
            # Use completions API
            response = await client.completions.create(
                model=model,
                prompt=full_prompt,
                temperature=self.temperature,
                max_tokens=1,
                logprobs=10,
            )

            choice = response.choices[0]
            if choice.logprobs and choice.logprobs.top_logprobs:
                return {
                    "top_logprobs": choice.logprobs.top_logprobs[0] if choice.logprobs.top_logprobs else {},
                    "token": choice.text.strip() if choice.text else "",
                }
        except Exception as e:
            # Log error but don't fail
            pass

        return {"top_logprobs": {}, "token": ""}

    def _extract_positive_prob(self, eval_response: dict) -> float:
        """Extract log probability of positive token from response."""
        top_logprobs = eval_response.get("top_logprobs", {})

        # Look for any positive token in the logprobs
        for token in self.positive_tokens:
            if token in top_logprobs:
                return top_logprobs[token]

        # If positive token not in top logprobs, use fallback
        return self.fallback_score

    def get_name(self) -> str:
        prompt_preview = self.eval_prompt[:25] + "..." if len(self.eval_prompt) > 25 else self.eval_prompt
        return f"SelfEvalScorer(prompt='{prompt_preview}')"


class CompositeScorer(Scorer):
    """
    Combine multiple scorers with configurable weights.

    Useful for combining PowerScorer with SelfEvalScorer to get
    both model confidence and self-evaluation signals.
    """

    def __init__(
        self,
        scorers: List[Tuple[Scorer, float]],
        use_length_penalty: bool = False,
        length_penalty: float = 0.6,
        use_ema: bool = False,
        ema_decay: float = 0.9,
    ):
        """
        Initialize CompositeScorer.

        Args:
            scorers: List of (scorer, weight) tuples
            use_length_penalty: Whether to apply length normalization (applied after combining)
            length_penalty: Exponent for length normalization
            use_ema: Whether to use EMA for cumulative scores
            ema_decay: Decay factor for EMA
        """
        super().__init__(use_length_penalty, length_penalty, use_ema, ema_decay)
        self.scorers = scorers

        # Normalize weights
        total_weight = sum(w for _, w in scorers)
        if total_weight > 0:
            self.normalized_weights = [(s, w / total_weight) for s, w in scorers]
        else:
            self.normalized_weights = [(s, 1.0 / len(scorers)) for s, _ in scorers]

    async def score_block(
        self,
        client: Any,
        prompt: str,
        prefix: str,
        block_text: str,
        block_tokens: List[str],
        block_log_p: List[float],
        sequence_id: int = 0,
    ) -> ScoreResult:
        """Score using weighted combination of all scorers."""
        # Run all scorers in parallel
        tasks = [
            scorer.score_block(
                client, prompt, prefix, block_text,
                block_tokens, block_log_p, sequence_id
            )
            for scorer, _ in self.scorers
        ]
        results = await asyncio.gather(*tasks)

        # Compute weighted score
        weighted_score = sum(
            result.score * weight
            for result, (_, weight) in zip(results, self.normalized_weights)
        )

        # Update cumulative (note: individual scorers also track their own cumulative)
        cumulative = self._update_cumulative(sequence_id, weighted_score)

        return ScoreResult(
            score=weighted_score,
            cumulative_score=cumulative,
            metadata={
                "component_scores": [r.score for r in results],
                "component_names": [s.get_name() for s, _ in self.scorers],
                "weights": [w for _, w in self.normalized_weights],
            }
        )

    def reset_sequence(self, sequence_id: int) -> None:
        """Reset cumulative tracking for a sequence in all scorers."""
        super().reset_sequence(sequence_id)
        for scorer, _ in self.scorers:
            scorer.reset_sequence(sequence_id)

    def reset_all(self) -> None:
        """Reset all cumulative tracking in all scorers."""
        super().reset_all()
        for scorer, _ in self.scorers:
            scorer.reset_all()

    def get_name(self) -> str:
        components = ", ".join(
            f"{s.get_name()}:{w:.2f}"
            for s, w in self.normalized_weights
        )
        return f"CompositeScorer([{components}])"


def create_scorer(
    scorer_type: str = "power",
    alpha: float = 4.0,
    eval_prompt: Optional[str] = None,
    positive_tokens: Optional[List[str]] = None,
    eval_temperature: float = 0.0,
    fallback_score: float = -10.0,
    power_weight: float = 0.7,
    self_eval_weight: float = 0.3,
    use_length_penalty: bool = False,
    length_penalty: float = 0.6,
    use_ema: bool = False,
    ema_decay: float = 0.9,
    model: Optional[str] = None,
) -> Scorer:
    """
    Factory function to create a scorer from parameters.

    Args:
        scorer_type: One of "power", "self_eval", or "composite"
        alpha: Power factor for PowerScorer
        eval_prompt: Evaluation prompt for SelfEvalScorer
        positive_tokens: Positive tokens for SelfEvalScorer
        eval_temperature: Temperature for SelfEvalScorer API calls
        fallback_score: Fallback score for SelfEvalScorer
        power_weight: Weight for PowerScorer in CompositeScorer
        self_eval_weight: Weight for SelfEvalScorer in CompositeScorer
        use_length_penalty: Whether to apply length normalization
        length_penalty: Exponent for length normalization
        use_ema: Whether to use EMA for cumulative scores
        ema_decay: Decay factor for EMA
        model: Optional model name for SelfEvalScorer

    Returns:
        Configured Scorer instance
    """
    if scorer_type == "power":
        return PowerScorer(
            alpha=alpha,
            use_length_penalty=use_length_penalty,
            length_penalty=length_penalty,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )
    elif scorer_type == "self_eval":
        return SelfEvalScorer(
            eval_prompt=eval_prompt,
            positive_tokens=positive_tokens,
            temperature=eval_temperature,
            use_length_penalty=use_length_penalty,
            length_penalty=length_penalty,
            use_ema=use_ema,
            ema_decay=ema_decay,
            fallback_score=fallback_score,
            model=model,
        )
    elif scorer_type == "composite":
        power_scorer = PowerScorer(
            alpha=alpha,
            use_length_penalty=False,  # Applied at composite level
            use_ema=False,  # Tracked at composite level
        )
        self_eval_scorer = SelfEvalScorer(
            eval_prompt=eval_prompt,
            positive_tokens=positive_tokens,
            temperature=eval_temperature,
            use_length_penalty=False,  # Applied at composite level
            use_ema=False,  # Tracked at composite level
            fallback_score=fallback_score,
            model=model,
        )
        return CompositeScorer(
            scorers=[
                (power_scorer, power_weight),
                (self_eval_scorer, self_eval_weight),
            ],
            use_length_penalty=use_length_penalty,
            length_penalty=length_penalty,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )
    else:
        raise ValueError(f"Unknown scorer type: {scorer_type}. Must be 'power', 'self_eval', or 'composite'")
