"""
Benchmark base classes and implementations for evaluating LLM reasoning strategies.

This module contains:
- Benchmark ABC: Abstract base class defining the benchmark interface
- HumanEvalBenchmark: Implementation for HumanEval code generation benchmark
- Utility functions for code extraction and evaluation
"""

import re
from abc import ABC, abstractmethod
from typing import Dict
from datasets import load_dataset


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
