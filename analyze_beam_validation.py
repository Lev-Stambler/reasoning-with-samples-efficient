#!/usr/bin/env python3
"""
Utility functions for analyzing beam validation results.
"""
from typing import List, Optional, Tuple
from src.benchmark_runner import Beam


def filter_by_validation(
    beams: List[Beam],
    require_yes: bool = True,
    max_entropy: Optional[float] = None,
    min_confidence: Optional[float] = None
) -> List[Beam]:
    """
    Filter beams based on validation criteria.
    
    Args:
        beams: List of Beam objects with validation data
        require_yes: If True, only keep beams with "Yes" validation
        max_entropy: Maximum allowed entropy (None = no filter)
        min_confidence: Minimum required confidence (None = no filter)
    
    Returns:
        Filtered list of beams
    """
    filtered = []
    
    for beam in beams:
        # Check validation response
        if require_yes and beam.validation_response != "Yes":
            continue
        
        # Check entropy threshold
        if max_entropy is not None and beam.validation_entropy is not None:
            if beam.validation_entropy > max_entropy:
                continue
        
        # Check confidence threshold
        if min_confidence is not None and beam.validation_confidence is not None:
            if beam.validation_confidence < min_confidence:
                continue
        
        filtered.append(beam)
    
    return filtered


def rank_by_validation(
    beams: List[Beam],
    entropy_weight: float = 1.0,
    confidence_weight: float = 1.0,
    response_bonus: float = 1.0
) -> List[Tuple[float, Beam]]:
    """
    Rank beams by validation quality.
    
    Score formula:
        validation_score = (response_bonus if "Yes" else 0) 
                         - entropy * entropy_weight 
                         + confidence * confidence_weight
    
    Args:
        beams: List of Beam objects
        entropy_weight: Weight for entropy penalty (higher = penalize uncertainty more)
        confidence_weight: Weight for confidence bonus
        response_bonus: Bonus for "Yes" response
    
    Returns:
        List of (validation_score, beam) tuples, sorted by score descending
    """
    scored = []
    
    for beam in beams:
        score = 0.0
        
        # Bonus for "Yes"
        if beam.validation_response == "Yes":
            score += response_bonus
        
        # Penalty for high entropy (uncertainty)
        if beam.validation_entropy is not None:
            score -= beam.validation_entropy * entropy_weight
        
        # Bonus for high confidence
        if beam.validation_confidence is not None:
            score += beam.validation_confidence * confidence_weight
        
        scored.append((score, beam))
    
    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def analyze_validation_stats(beams: List[Beam]) -> dict:
    """
    Compute statistics about validation results.
    
    Returns:
        Dictionary with statistics:
        - num_yes: Number of "Yes" validations
        - num_no: Number of "No" validations
        - avg_entropy: Average entropy
        - avg_confidence: Average confidence
        - avg_entropy_yes: Average entropy for "Yes" responses
        - avg_entropy_no: Average entropy for "No" responses
    """
    yes_beams = [b for b in beams if b.validation_response == "Yes"]
    no_beams = [b for b in beams if b.validation_response == "No"]
    
    entropies = [b.validation_entropy for b in beams if b.validation_entropy is not None]
    confidences = [b.validation_confidence for b in beams if b.validation_confidence is not None]
    
    entropies_yes = [b.validation_entropy for b in yes_beams if b.validation_entropy is not None]
    entropies_no = [b.validation_entropy for b in no_beams if b.validation_entropy is not None]
    
    return {
        "num_yes": len(yes_beams),
        "num_no": len(no_beams),
        "yes_rate": len(yes_beams) / len(beams) if beams else 0,
        "avg_entropy": sum(entropies) / len(entropies) if entropies else None,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
        "avg_entropy_yes": sum(entropies_yes) / len(entropies_yes) if entropies_yes else None,
        "avg_entropy_no": sum(entropies_no) / len(entropies_no) if entropies_no else None,
        "min_entropy": min(entropies) if entropies else None,
        "max_entropy": max(entropies) if entropies else None,
    }


def print_validation_report(beams: List[Beam], top_n: int = 5):
    """
    Print a detailed validation report for beams.
    
    Args:
        beams: List of Beam objects with validation data
        top_n: Number of top/bottom beams to show
    """
    if not beams:
        print("No beams to analyze")
        return
    
    print("=" * 80)
    print("BEAM VALIDATION REPORT")
    print("=" * 80)
    
    # Overall stats
    stats = analyze_validation_stats(beams)
    print("\nOverall Statistics:")
    print(f"  Total beams: {len(beams)}")
    print(f"  Yes: {stats['num_yes']} ({stats['yes_rate']:.1%})")
    print(f"  No: {stats['num_no']} ({1-stats['yes_rate']:.1%})")
    print(f"  Avg entropy: {stats['avg_entropy']:.3f}" if stats['avg_entropy'] else "  Avg entropy: N/A")
    print(f"  Avg confidence: {stats['avg_confidence']:.3f}" if stats['avg_confidence'] else "  Avg confidence: N/A")
    
    if stats['avg_entropy_yes'] is not None:
        print(f"  Avg entropy (Yes): {stats['avg_entropy_yes']:.3f}")
    if stats['avg_entropy_no'] is not None:
        print(f"  Avg entropy (No): {stats['avg_entropy_no']:.3f}")
    
    # Rank by validation quality
    ranked = rank_by_validation(beams)
    
    print(f"\nTop {min(top_n, len(ranked))} Best Validated Beams:")
    print("-" * 80)
    for i, (score, beam) in enumerate(ranked[:top_n], 1):
        entropy_str = f"{beam.validation_entropy:.3f}" if beam.validation_entropy else "N/A"
        conf_str = f"{beam.validation_confidence:.3f}" if beam.validation_confidence else "N/A"
        text_preview = beam.text[:60] + "..." if len(beam.text) > 60 else beam.text
        print(f"{i}. Score: {score:.3f} | {beam.validation_response} (ent={entropy_str}, conf={conf_str})")
        print(f"   {text_preview}")
        print()
    
    print(f"\nBottom {min(top_n, len(ranked))} Worst Validated Beams:")
    print("-" * 80)
    for i, (score, beam) in enumerate(ranked[-top_n:][::-1], 1):
        entropy_str = f"{beam.validation_entropy:.3f}" if beam.validation_entropy else "N/A"
        conf_str = f"{beam.validation_confidence:.3f}" if beam.validation_confidence else "N/A"
        text_preview = beam.text[:60] + "..." if len(beam.text) > 60 else beam.text
        print(f"{i}. Score: {score:.3f} | {beam.validation_response} (ent={entropy_str}, conf={conf_str})")
        print(f"   {text_preview}")
        print()
    
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    # This would be used with actual beams from beam search
    print("This module provides utilities for analyzing beam validation results.")
    print("\nExample usage:")
    print("""
    from analyze_beam_validation import filter_by_validation, rank_by_validation
    
    # Filter beams with confident "Yes" responses
    good_beams = filter_by_validation(
        beams, 
        require_yes=True,
        max_entropy=0.5,  # Only very certain answers
        min_confidence=0.9
    )
    
    # Rank beams by validation quality
    ranked = rank_by_validation(beams, entropy_weight=2.0)
    best_beam = ranked[0][1]
    
    # Print detailed report
    print_validation_report(beams, top_n=10)
    """)
