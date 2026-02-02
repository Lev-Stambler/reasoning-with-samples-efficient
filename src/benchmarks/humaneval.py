"""
HumanEval benchmark implementation.
"""

import re
from typing import Dict
from datasets import load_dataset
from .base import Benchmark

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
