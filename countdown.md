# Countdown Numbers Environment Implementation Guide

> **For:** Coding assistant implementing within [PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)  
> **Goal:** Full implementation from dataset → environment → eval → RL training on 3B model

---

## Table of Contents
1. [Game Overview](#1-game-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset Creation](#3-dataset-creation)
4. [Environment Implementation](#4-environment-implementation)
5. [Rubric & Reward Shaping](#5-rubric--reward-shaping)
6. [Evaluation Setup](#6-evaluation-setup)
7. [RL Training Pipeline](#7-rl-training-pipeline)
8. [Testing & Validation](#8-testing--validation)

---

## 1. Game Overview

### Rules (UK TV Show "Countdown")
- **Given:** 6 numbers selected from:
  - Large numbers: `{25, 50, 75, 100}`
  - Small numbers: `{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}` (each can appear twice)
- **Target:** A random 3-digit number (101-999)
- **Goal:** Combine the 6 numbers using `+`, `-`, `*`, `/` to reach the target exactly (or as close as possible)
- **Constraints:**
  - Each given number can only be used **once**
  - All intermediate results must be **positive integers** (no fractions, no negatives)
  - Division must be exact (no remainders)
  - Not all numbers need to be used

### Example
```
Target: 952
Numbers: 25, 50, 75, 100, 3, 6

Solution: 100 + 6 = 106
          106 * 3 = 318  
          ❌ This path doesn't work...

Better: 75 + 50 = 125
        125 - 3 = 122  
        ❌ Still not right...

Correct: 100 * 6 = 600
         600 + 75 = 675
         ❌ Nope...

Actually: (100 + 3) * (75 - 50 / 25 * 6) = ...
         Let's compute: 50/25=2, 2*6=12, 75-12=63, 100+3=103, 103*63 = 6489 ❌

Real solution: 100 - 6 = 94
               94 + 3 = 97  
               ❌

Simpler example - Target: 127, Numbers: 100, 25, 10, 5, 2, 1
Solution: 100 + 25 + 2 = 127 ✓
```

### Why It's Good for RL
- **Deterministic verification** - arithmetic is checkable
- **Rich reward shaping** - distance to target, solution elegance
- **Variable difficulty** - easy (addition only) to hard (complex expressions)
- **Procedurally generatable** - infinite training data
- **Multi-step reasoning** - requires search/planning

---

## 2. Project Structure

Initialize the environment:
```bash
cd /path/to/verifiers
uv run vf-init countdown_numbers
```

This creates:
```
environments/countdown_numbers/
├── countdown_numbers.py    # Main implementation
├── pyproject.toml          # Dependencies
└── README.md               # Documentation
```

Final structure after implementation:
```
environments/countdown_numbers/
├── countdown_numbers.py    # Environment + rubric
├── generator.py            # Puzzle generator with solver
├── solver.py               # Reverse Polish Notation solver/verifier
├── pyproject.toml
├── README.md
└── tests/
    └── test_countdown.py
```

---

## 3. Dataset Creation

### 3.1 Puzzle Generator (`generator.py`)

```python
"""
Countdown Numbers puzzle generator with guaranteed solvability.
"""
import random
from dataclasses import dataclass
from typing import Optional
from itertools import permutations, product


@dataclass
class CountdownPuzzle:
    target: int
    numbers: list[int]
    solution: Optional[str] = None  # Optional: a known solution
    difficulty: str = "medium"      # easy, medium, hard


LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def generate_numbers(num_large: int = 2) -> list[int]:
    """
    Generate 6 numbers for a Countdown puzzle.
    
    Args:
        num_large: How many large numbers (0-4). Default 2 is standard.
    
    Returns:
        List of 6 numbers (sorted descending for consistency)
    """
    num_large = max(0, min(4, num_large))
    num_small = 6 - num_large
    
    large = random.sample(LARGE_NUMBERS, num_large)
    # Small numbers: each can appear at most twice
    small_pool = SMALL_NUMBERS * 2
    small = random.sample(small_pool, num_small)
    
    numbers = large + small
    return sorted(numbers, reverse=True)


def evaluate_rpn(tokens: list) -> Optional[int]:
    """
    Evaluate Reverse Polish Notation expression.
    Returns None if invalid (division by zero, non-integer, negative).
    """
    stack = []
    for token in tokens:
        if isinstance(token, int):
            stack.append(token)
        else:
            if len(stack) < 2:
                return None
            b, a = stack.pop(), stack.pop()
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
                if result <= 0:
                    return None
            elif token == '*':
                result = a * b
            elif token == '/':
                if b == 0 or a % b != 0:
                    return None
                result = a // b
            else:
                return None
            stack.append(result)
    
    return stack[0] if len(stack) == 1 else None


def solve_countdown(numbers: list[int], target: int, max_solutions: int = 1) -> list[str]:
    """
    Find solutions using brute-force search over all combinations.
    
    This is a simplified solver - for production, use more efficient algorithms.
    Returns list of solution strings in infix notation.
    """
    solutions = []
    ops = ['+', '-', '*', '/']
    
    # Try all subsets of numbers (2 to 6 numbers)
    for r in range(2, len(numbers) + 1):
        for perm in permutations(numbers, r):
            # Try all operator combinations
            for op_combo in product(ops, repeat=r-1):
                # Build RPN: n1 n2 op1 n3 op2 ... (left-to-right evaluation)
                # This is a simplification - full solver would try all tree structures
                rpn = [perm[0], perm[1], op_combo[0]]
                for i in range(2, r):
                    rpn.extend([perm[i], op_combo[i-1]])
                
                result = evaluate_rpn(rpn)
                if result == target:
                    # Convert to infix for readability
                    infix = rpn_to_infix(rpn)
                    if infix not in solutions:
                        solutions.append(infix)
                        if len(solutions) >= max_solutions:
                            return solutions
    
    return solutions


def rpn_to_infix(tokens: list) -> str:
    """Convert RPN to infix notation string."""
    stack = []
    for token in tokens:
        if isinstance(token, int):
            stack.append(str(token))
        else:
            b, a = stack.pop(), stack.pop()
            stack.append(f"({a} {token} {b})")
    return stack[0] if stack else ""


def find_closest_achievable(numbers: list[int], target: int) -> tuple[int, str]:
    """
    Find the closest achievable result to target.
    Returns (closest_value, solution_string).
    """
    best_diff = float('inf')
    best_result = 0
    best_solution = ""
    
    ops = ['+', '-', '*', '/']
    
    for r in range(2, len(numbers) + 1):
        for perm in permutations(numbers, r):
            for op_combo in product(ops, repeat=r-1):
                rpn = [perm[0], perm[1], op_combo[0]]
                for i in range(2, r):
                    rpn.extend([perm[i], op_combo[i-1]])
                
                result = evaluate_rpn(rpn)
                if result is not None:
                    diff = abs(result - target)
                    if diff < best_diff:
                        best_diff = diff
                        best_result = result
                        best_solution = rpn_to_infix(rpn)
                        if diff == 0:
                            return best_result, best_solution
    
    return best_result, best_solution


def generate_puzzle(
    difficulty: str = "medium",
    ensure_solvable: bool = True
) -> CountdownPuzzle:
    """
    Generate a Countdown puzzle.
    
    Args:
        difficulty: "easy", "medium", or "hard"
        ensure_solvable: If True, only return puzzles with exact solutions
    
    Returns:
        CountdownPuzzle with target, numbers, and optional solution
    """
    difficulty_config = {
        "easy": {"num_large": 0, "target_range": (101, 200)},
        "medium": {"num_large": 2, "target_range": (101, 999)},
        "hard": {"num_large": 4, "target_range": (500, 999)},
    }
    
    config = difficulty_config.get(difficulty, difficulty_config["medium"])
    
    max_attempts = 100
    for _ in range(max_attempts):
        numbers = generate_numbers(config["num_large"])
        target = random.randint(*config["target_range"])
        
        if ensure_solvable:
            solutions = solve_countdown(numbers, target, max_solutions=1)
            if solutions:
                return CountdownPuzzle(
                    target=target,
                    numbers=numbers,
                    solution=solutions[0],
                    difficulty=difficulty
                )
        else:
            return CountdownPuzzle(
                target=target,
                numbers=numbers,
                difficulty=difficulty
            )
    
    # Fallback: generate from solution (guaranteed solvable)
    return generate_puzzle_from_solution(difficulty)


def generate_puzzle_from_solution(difficulty: str = "medium") -> CountdownPuzzle:
    """
    Generate puzzle by working backwards from a valid solution.
    Guarantees solvability.
    """
    difficulty_config = {
        "easy": {"num_large": 0, "ops": ['+'], "steps": 2},
        "medium": {"num_large": 2, "ops": ['+', '-', '*'], "steps": 3},
        "hard": {"num_large": 4, "ops": ['+', '-', '*', '/'], "steps": 4},
    }
    
    config = difficulty_config.get(difficulty, difficulty_config["medium"])
    numbers = generate_numbers(config["num_large"])
    
    # Pick random subset and build expression
    num_to_use = min(config["steps"] + 1, len(numbers))
    subset = random.sample(numbers, num_to_use)
    
    # Build expression left-to-right
    result = subset[0]
    expression_parts = [str(subset[0])]
    
    for i in range(1, len(subset)):
        op = random.choice(config["ops"])
        n = subset[i]
        
        if op == '+':
            result = result + n
        elif op == '-':
            if result > n:
                result = result - n
            else:
                op = '+'
                result = result + n
        elif op == '*':
            result = result * n
        elif op == '/':
            if result % n == 0 and n != 0:
                result = result // n
            else:
                op = '+'
                result = result + n
        
        expression_parts.append(f"{op} {n}")
    
    solution = " ".join(expression_parts)
    
    return CountdownPuzzle(
        target=result,
        numbers=numbers,
        solution=solution,
        difficulty=difficulty
    )


def generate_dataset(
    num_examples: int = 1000,
    difficulty_distribution: dict = None,
    seed: int = 42
) -> list[dict]:
    """
    Generate a dataset of Countdown puzzles.
    
    Returns list of dicts ready for HuggingFace Dataset.
    """
    random.seed(seed)
    
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.2, "medium": 0.6, "hard": 0.2}
    
    dataset = []
    for i in range(num_examples):
        # Sample difficulty
        r = random.random()
        cumsum = 0
        difficulty = "medium"
        for diff, prob in difficulty_distribution.items():
            cumsum += prob
            if r < cumsum:
                difficulty = diff
                break
        
        puzzle = generate_puzzle(difficulty=difficulty, ensure_solvable=True)
        
        # Format as chat message
        prompt_text = f"""You are playing the Countdown Numbers game.

Target: {puzzle.target}
Available numbers: {', '.join(map(str, puzzle.numbers))}

Rules:
- Use +, -, *, / to reach the target exactly
- Each number can only be used once
- All intermediate results must be positive integers
- Division must be exact (no remainders)
- You don't have to use all numbers

Show your working step by step, then provide your final answer in the format:
<answer>YOUR_EXPRESSION = {puzzle.target}</answer>

For example: <answer>(100 + 25) * 4 = 500</answer>"""

        dataset.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "answer": str(puzzle.target),
            "info": {
                "target": puzzle.target,
                "numbers": puzzle.numbers,
                "solution": puzzle.solution,
                "difficulty": puzzle.difficulty
            }
        })
    
    return dataset


if __name__ == "__main__":
    # Test generation
    for diff in ["easy", "medium", "hard"]:
        puzzle = generate_puzzle(difficulty=diff)
        print(f"\n{diff.upper()}: Target={puzzle.target}, Numbers={puzzle.numbers}")
        print(f"  Solution: {puzzle.solution}")
```

### 3.2 Solution Verifier (`solver.py`)

```python
"""
Countdown Numbers solution verifier.
Parses and validates player solutions.
"""
import re
from typing import Optional, Tuple


def parse_expression(expr: str) -> Optional[int]:
    """
    Safely evaluate a mathematical expression.
    Only allows +, -, *, /, parentheses, and integers.
    
    Returns the result or None if invalid.
    """
    # Clean the expression
    expr = expr.strip()
    
    # Remove the "= result" part if present
    if '=' in expr:
        expr = expr.split('=')[0].strip()
    
    # Validate characters (only digits, operators, parens, spaces)
    if not re.match(r'^[\d\s\+\-\*\/\(\)]+$', expr):
        return None
    
    try:
        # Use Python's eval with restricted globals
        # This is safe because we've validated the characters
        result = eval(expr, {"__builtins__": {}}, {})
        
        # Check it's a positive integer
        if isinstance(result, (int, float)):
            if result == int(result) and result > 0:
                return int(result)
        return None
    except:
        return None


def extract_numbers_used(expr: str) -> list[int]:
    """Extract all numbers used in an expression."""
    # Remove the "= result" part if present
    if '=' in expr:
        expr = expr.split('=')[0]
    
    numbers = re.findall(r'\d+', expr)
    return [int(n) for n in numbers]


def verify_solution(
    expr: str,
    target: int,
    available_numbers: list[int]
) -> Tuple[bool, str, Optional[int]]:
    """
    Verify a Countdown solution.
    
    Args:
        expr: The expression to verify (e.g., "(100 + 25) * 4")
        target: The target number to reach
        available_numbers: The numbers that were available
    
    Returns:
        (is_valid, message, result)
        - is_valid: True if solution is correct
        - message: Explanation of result or error
        - result: The computed value (or None if invalid)
    """
    # Parse and evaluate
    result = parse_expression(expr)
    
    if result is None:
        return False, "Invalid expression (could not parse or non-integer result)", None
    
    # Check numbers used are available
    numbers_used = extract_numbers_used(expr)
    available_copy = available_numbers.copy()
    
    for n in numbers_used:
        if n in available_copy:
            available_copy.remove(n)
        else:
            return False, f"Number {n} not available or used too many times", result
    
    # Check if target is reached
    if result == target:
        return True, f"Correct! {expr} = {target}", result
    else:
        diff = abs(result - target)
        return False, f"Result is {result}, off by {diff}", result


def validate_intermediate_steps(expr: str) -> Tuple[bool, str]:
    """
    Check that all intermediate results are positive integers.
    This is a simplified check - full validation would need expression tree parsing.
    """
    # For now, we just check the final result
    # A more thorough implementation would parse the expression tree
    # and verify each intermediate step
    result = parse_expression(expr)
    if result is None:
        return False, "Invalid intermediate result"
    return True, "All steps valid"


def score_solution(
    expr: str,
    target: int,
    available_numbers: list[int]
) -> float:
    """
    Score a solution from 0.0 to 1.0.
    
    Scoring:
    - Exact match: 1.0
    - Off by 1-5: 0.7 - 0.9
    - Off by 6-10: 0.5 - 0.7
    - Off by 11-20: 0.3 - 0.5
    - Off by more: 0.0 - 0.3 (proportional)
    - Invalid: 0.0
    """
    is_valid, msg, result = verify_solution(expr, target, available_numbers)
    
    if result is None:
        return 0.0
    
    if is_valid:
        return 1.0
    
    diff = abs(result - target)
    
    if diff <= 5:
        return 0.9 - (diff - 1) * 0.05  # 0.9 to 0.7
    elif diff <= 10:
        return 0.7 - (diff - 5) * 0.04  # 0.7 to 0.5
    elif diff <= 20:
        return 0.5 - (diff - 10) * 0.02  # 0.5 to 0.3
    else:
        # Exponential decay for larger differences
        return max(0.0, 0.3 * (0.95 ** (diff - 20)))


if __name__ == "__main__":
    # Test verification
    test_cases = [
        ("100 + 25 + 2", 127, [100, 25, 10, 5, 2, 1]),
        ("(100 + 3) * 9", 927, [100, 75, 50, 25, 9, 3]),
        ("100 + 200", 300, [100, 25, 10, 5, 2, 1]),  # 200 not available
        ("invalid", 100, [100, 25]),
    ]
    
    for expr, target, numbers in test_cases:
        is_valid, msg, result = verify_solution(expr, target, numbers)
        score = score_solution(expr, target, numbers)
        print(f"\nExpr: {expr}")
        print(f"  Target: {target}, Numbers: {numbers}")
        print(f"  Valid: {is_valid}, Result: {result}")
        print(f"  Message: {msg}")
        print(f"  Score: {score:.2f}")
```

### 3.3 Create HuggingFace Dataset

```python
# scripts/create_countdown_dataset.py
"""
Create and upload Countdown Numbers dataset to HuggingFace.
"""
import sys
sys.path.insert(0, 'environments/countdown_numbers')

from datasets import Dataset
from generator import generate_dataset

def main():
    # Generate datasets
    train_data = generate_dataset(num_examples=10000, seed=42)
    val_data = generate_dataset(num_examples=500, seed=123)
    test_data = generate_dataset(num_examples=500, seed=456)
    
    # Convert to HuggingFace datasets
    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)
    test_ds = Dataset.from_list(test_data)
    
    # Save locally first
    train_ds.save_to_disk("data/countdown_numbers/train")
    val_ds.save_to_disk("data/countdown_numbers/val")
    test_ds.save_to_disk("data/countdown_numbers/test")
    
    print(f"Created dataset:")
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Val: {len(val_ds)} examples")
    print(f"  Test: {len(test_ds)} examples")
    print(f"\nSample:")
    print(train_ds[0])
    
    # Optional: Upload to HuggingFace Hub
    # from huggingface_hub import login
    # login()
    # train_ds.push_to_hub("your-username/countdown-numbers", split="train")
    # val_ds.push_to_hub("your-username/countdown-numbers", split="validation")
    # test_ds.push_to_hub("your-username/countdown-numbers", split="test")


if __name__ == "__main__":
    main()
```

---

## 4. Environment Implementation

### 4.1 Main Environment (`countdown_numbers.py`)

```python
"""
Countdown Numbers Environment for verifiers.

A mathematical puzzle game where players combine numbers 
using arithmetic operations to reach a target value.
"""
import re
from typing import Optional
from datasets import Dataset, load_dataset, load_from_disk

import verifiers as vf
from verifiers.parsers import XMLParser

# Import local modules
from .generator import generate_dataset, generate_puzzle
from .solver import verify_solution, score_solution, parse_expression, extract_numbers_used


class CountdownParser(XMLParser):
    """Parser for Countdown answers in <answer>...</answer> tags."""
    
    def __init__(self):
        super().__init__(
            fields=["answer"],
            required_fields=["answer"]
        )
    
    def parse_answer(self, text: str) -> Optional[str]:
        """Extract the expression from <answer> tags."""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None


def load_environment(
    num_examples: int = -1,
    difficulty: str = "mixed",  # "easy", "medium", "hard", or "mixed"
    dataset_path: Optional[str] = None,
    seed: int = 42,
) -> vf.Environment:
    """
    Load the Countdown Numbers environment.
    
    Args:
        num_examples: Number of examples (-1 for all)
        difficulty: Difficulty level or "mixed" for distribution
        dataset_path: Path to pre-generated dataset (optional)
        seed: Random seed for generation
    
    Returns:
        Configured verifiers Environment
    """
    
    # Load or generate dataset
    if dataset_path:
        try:
            dataset = load_from_disk(dataset_path)
        except:
            dataset = load_dataset(dataset_path, split="train")
    else:
        # Generate on-the-fly
        if difficulty == "mixed":
            dist = {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        else:
            dist = {difficulty: 1.0}
        
        n = num_examples if num_examples > 0 else 1000
        data = generate_dataset(num_examples=n, difficulty_distribution=dist, seed=seed)
        dataset = Dataset.from_list(data)
    
    # Slice if needed
    if num_examples > 0 and num_examples < len(dataset):
        dataset = dataset.select(range(num_examples))
    
    # Create parser
    parser = CountdownParser()
    
    # Define reward functions
    def correctness_reward(completion, answer, info) -> float:
        """
        Main reward: how close is the answer to the target?
        1.0 for exact match, scaled down for near misses.
        """
        # Get the last assistant message
        last_msg = completion[-1]["content"] if completion else ""
        
        # Parse the answer
        parsed = parser.parse_answer(last_msg)
        if parsed is None:
            return 0.0
        
        target = info["target"]
        numbers = info["numbers"]
        
        return score_solution(parsed, target, numbers)
    
    def format_reward(completion, **kwargs) -> float:
        """
        Format adherence: did they use the <answer> tag correctly?
        """
        last_msg = completion[-1]["content"] if completion else ""
        
        # Check for answer tag
        if "<answer>" not in last_msg.lower() or "</answer>" not in last_msg.lower():
            return 0.0
        
        # Check it contains an expression
        parsed = parser.parse_answer(last_msg)
        if parsed is None:
            return 0.0
        
        # Check it looks like a math expression
        if not re.search(r'\d+\s*[\+\-\*\/]', parsed):
            return 0.5
        
        return 1.0
    
    def efficiency_reward(completion, info, **kwargs) -> float:
        """
        Bonus for elegant solutions (fewer numbers used).
        Only applies if the answer is correct.
        """
        last_msg = completion[-1]["content"] if completion else ""
        parsed = parser.parse_answer(last_msg)
        
        if parsed is None:
            return 0.0
        
        target = info["target"]
        numbers = info["numbers"]
        
        is_valid, _, result = verify_solution(parsed, target, numbers)
        
        if not is_valid:
            return 0.0
        
        # Count numbers used
        nums_used = extract_numbers_used(parsed)
        num_count = len(nums_used)
        
        # Reward for using fewer numbers (max 6)
        # 2 numbers: 1.0, 3: 0.8, 4: 0.6, 5: 0.4, 6: 0.2
        return max(0.0, 1.0 - (num_count - 2) * 0.2)
    
    def reasoning_reward(completion, **kwargs) -> float:
        """
        Reward for showing work (step-by-step reasoning).
        """
        last_msg = completion[-1]["content"] if completion else ""
        
        # Count intermediate calculations shown
        # Look for patterns like "100 + 25 = 125"
        calc_pattern = r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+'
        calculations = re.findall(calc_pattern, last_msg)
        
        # Also count lines that look like reasoning
        lines = last_msg.split('\n')
        reasoning_lines = sum(1 for line in lines if '=' in line or any(op in line for op in ['+', '-', '*', '/']))
        
        # Reward for showing work (at least 2 steps for full credit)
        if len(calculations) >= 2 or reasoning_lines >= 3:
            return 1.0
        elif len(calculations) >= 1 or reasoning_lines >= 2:
            return 0.5
        return 0.0
    
    # Create rubric with weighted rewards
    rubric = vf.Rubric(
        funcs=[correctness_reward, format_reward, efficiency_reward, reasoning_reward],
        weights=[1.0, 0.2, 0.1, 0.1]  # Correctness is most important
    )
    
    # Create environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        rubric=rubric,
        parser=parser,
    )
    
    return env


# Convenience function for interactive play
def play_countdown(target: int = None, numbers: list[int] = None):
    """Play a single Countdown puzzle interactively."""
    if target is None or numbers is None:
        puzzle = generate_puzzle(difficulty="medium")
        target = puzzle.target
        numbers = puzzle.numbers
        print(f"Solution hint: {puzzle.solution}")
    
    print(f"\n{'='*50}")
    print(f"TARGET: {target}")
    print(f"NUMBERS: {', '.join(map(str, numbers))}")
    print(f"{'='*50}\n")
    
    while True:
        expr = input("Your answer (or 'quit'): ").strip()
        if expr.lower() == 'quit':
            break
        
        is_valid, msg, result = verify_solution(expr, target, numbers)
        score = score_solution(expr, target, numbers)
        print(f"  {msg}")
        print(f"  Score: {score:.2f}")
        
        if is_valid:
            print("  🎉 Congratulations!")
            break


if __name__ == "__main__":
    play_countdown()
```

### 4.2 Package Configuration (`pyproject.toml`)

```toml
[project]
name = "countdown_numbers"
version = "0.1.0"
description = "Countdown Numbers mathematical puzzle environment for RL training"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "verifiers>=0.1.8",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 4.3 Module Init (`__init__.py`)

```python
"""Countdown Numbers Environment."""
from .countdown_numbers import load_environment, CountdownParser, play_countdown
from .generator import generate_puzzle, generate_dataset, CountdownPuzzle
from .solver import verify_solution, score_solution, parse_expression

__all__ = [
    "load_environment",
    "CountdownParser", 
    "play_countdown",
    "generate_puzzle",
    "generate_dataset",
    "CountdownPuzzle",
    "verify_solution",
    "score_solution",
    "parse_expression",
]
```

---

## 5. Rubric & Reward Shaping

### Reward Breakdown

| Reward Function | Weight | Description |
|-----------------|--------|-------------|
| `correctness_reward` | 1.0 | Core reward - how close to target (1.0 = exact, decays with distance) |
| `format_reward` | 0.2 | Did they use `<answer>` tags correctly? |
| `efficiency_reward` | 0.1 | Bonus for using fewer numbers (elegant solutions) |
| `reasoning_reward` | 0.1 | Bonus for showing step-by-step work |

### Reward Shaping Philosophy

The scoring is designed to:
1. **Prioritize correctness** - exact answers get full reward
2. **Encourage near-misses** - being off by 1-10 still gets partial credit (unlike binary 0/1)
3. **Reward format adherence** - model learns the output structure early
4. **Incentivize elegance** - simpler solutions are more interpretable

### Customizing Rewards

```python
# For curriculum learning, you might start with:
rubric = vf.Rubric(
    funcs=[correctness_reward, format_reward],
    weights=[0.5, 0.5]  # Equal weight initially
)

# Then shift to:
rubric = vf.Rubric(
    funcs=[correctness_reward, format_reward],
    weights=[1.0, 0.1]  # Focus on correctness
)
```

---

## 6. Evaluation Setup

### 6.1 Install Environment

```bash
cd /path/to/verifiers

# Install the environment
uv run vf-install countdown_numbers

# Verify installation
python -c "from verifiers import load_environment; env = load_environment('countdown_numbers'); print(f'Loaded {len(env.dataset)} examples')"
```

### 6.2 Run Evaluation

```bash
# Quick eval with GPT-4
uv run vf-eval countdown_numbers -m gpt-4.1-mini -n 20 -r 3

# Full eval with local model
uv run vf-eval countdown_numbers \
    -m Qwen/Qwen3-4B-Instruct \
    --api-base-url http://localhost:8000/v1 \
    -n 100 -r 1 -c 32
```

### 6.3 Programmatic Evaluation

```python
"""Evaluate Countdown Numbers environment."""
import asyncio
from openai import AsyncOpenAI
import verifiers as vf

async def main():
    # Load environment
    env = vf.load_environment("countdown_numbers", num_examples=50, difficulty="medium")
    
    # Setup client
    client = AsyncOpenAI()
    
    # Run evaluation
    results = await env.evaluate(
        client=client,
        model="gpt-4.1-mini",
        num_examples=50,
        rollouts_per_example=3,
        temperature=0.7,
        max_tokens=1024,
    )
    
    # Analyze results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    
    scores = [r["reward"] for r in results]
    print(f"Mean reward: {sum(scores)/len(scores):.3f}")
    print(f"Exact solves: {sum(1 for s in scores if s >= 0.9)}/{len(scores)}")
    
    # Show some examples
    print(f"\n{'='*50}")
    print("SAMPLE COMPLETIONS")
    print(f"{'='*50}")
    
    for i, r in enumerate(results[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Target: {r['info']['target']}")
        print(f"Numbers: {r['info']['numbers']}")
        print(f"Completion: {r['completion'][-1]['content'][:200]}...")
        print(f"Reward: {r['reward']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. RL Training Pipeline

### 7.1 Training Configuration

Create `configs/vf-rl/countdown_numbers.toml`:

```toml
[environment]
name = "countdown_numbers"

[environment.args]
num_examples = -1
difficulty = "mixed"

[model]
name = "Qwen/Qwen3-1.7B"  # Or your 3B model
# For 3B, use something like:
# name = "microsoft/phi-3.5-mini-instruct"  
# name = "Qwen/Qwen2.5-3B-Instruct"

[training]
# Basic settings
num_train_steps = 2000
batch_size = 32
gradient_accumulation_steps = 4
learning_rate = 1e-5
warmup_steps = 100

# RL-specific
rollouts_per_prompt = 4
max_new_tokens = 512
temperature = 0.8

# Regularization
kl_coef = 0.05
clip_range = 0.2

[training.lora]
# Use LoRA for memory efficiency on smaller GPUs
enabled = true
r = 16
alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
dropout = 0.05

[inference]
# vLLM server settings
tensor_parallel_size = 1
max_model_len = 2048
gpu_memory_utilization = 0.85

[logging]
project = "countdown-numbers-rl"
report_to = "wandb"
log_every = 10
eval_every = 200
save_every = 500
```

### 7.2 Training with `vf.RLTrainer`

```bash
# Setup (creates config templates)
uv run vf-setup

# Start training
uv run vf-rl @ configs/vf-rl/countdown_numbers.toml
```

### 7.3 Training with `prime-rl` (Recommended for Scale)

```bash
# Setup prime-rl
uv run vf-setup --prime-rl

# Edit configs/prime-rl/countdown_numbers.toml as needed
# Then:
uv run prime-rl @ configs/prime-rl/countdown_numbers.toml
```

### 7.4 Custom Training Script

For more control, create a custom training script:

```python
"""
Custom RL training script for Countdown Numbers.
"""
import verifiers as vf
from verifiers.rl import RLTrainer, RLConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load environment
    env = vf.load_environment(
        "countdown_numbers",
        num_examples=5000,
        difficulty="mixed"
    )
    
    # Load model
    model_name = "Qwen/Qwen2.5-3B-Instruct"  # Your 3B model
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Configure training
    config = RLConfig(
        # Model
        model_name=model_name,
        
        # Training
        num_train_steps=2000,
        batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        
        # RL
        rollouts_per_prompt=4,
        max_new_tokens=512,
        temperature=0.7,
        
        # Regularization
        kl_coef=0.05,
        
        # LoRA (for memory efficiency)
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        
        # Logging
        output_dir="outputs/countdown_numbers",
        logging_steps=10,
        eval_steps=200,
        save_steps=500,
        report_to="wandb",
        run_name="countdown-numbers-3b",
    )
    
    # Create trainer
    trainer = RLTrainer(
        model=model,
        tokenizer=tokenizer,
        env=env,
        config=config,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    trainer.save_model("outputs/countdown_numbers/final")


if __name__ == "__main__":
    main()
```

### 7.5 Curriculum Learning Strategy

For faster convergence, implement curriculum learning:

```python
"""Curriculum learning for Countdown Numbers."""
import verifiers as vf

def create_curriculum_envs():
    """Create environments for curriculum stages."""
    return {
        # Stage 1: Easy (addition only, small targets)
        "stage1": vf.load_environment(
            "countdown_numbers",
            difficulty="easy",
            num_examples=2000,
        ),
        # Stage 2: Medium
        "stage2": vf.load_environment(
            "countdown_numbers", 
            difficulty="medium",
            num_examples=3000,
        ),
        # Stage 3: Hard
        "stage3": vf.load_environment(
            "countdown_numbers",
            difficulty="hard", 
            num_examples=3000,
        ),
        # Stage 4: Mixed (final)
        "stage4": vf.load_environment(
            "countdown_numbers",
            difficulty="mixed",
            num_examples=5000,
        ),
    }

# Training schedule:
# - Stage 1: 500 steps
# - Stage 2: 700 steps  
# - Stage 3: 800 steps
# - Stage 4: 1000 steps (fine-tuning)
```

---

## 8. Testing & Validation

### 8.1 Unit Tests (`tests/test_countdown.py`)

```python
"""Tests for Countdown Numbers environment."""
import pytest
from countdown_numbers import (
    load_environment,
    generate_puzzle,
    verify_solution,
    score_solution,
    parse_expression,
)


class TestSolver:
    def test_parse_simple_expression(self):
        assert parse_expression("100 + 25") == 125
        assert parse_expression("100 - 25") == 75
        assert parse_expression("100 * 4") == 400
        assert parse_expression("100 / 4") == 25
    
    def test_parse_complex_expression(self):
        assert parse_expression("(100 + 25) * 4") == 500
        assert parse_expression("100 + 25 * 4") == 200
    
    def test_parse_invalid(self):
        assert parse_expression("invalid") is None
        assert parse_expression("100 / 3") is None  # Non-integer
        assert parse_expression("5 - 10") is None   # Negative
    
    def test_verify_valid_solution(self):
        is_valid, msg, result = verify_solution(
            "100 + 25 + 2",
            127,
            [100, 25, 10, 5, 2, 1]
        )
        assert is_valid
        assert result == 127
    
    def test_verify_invalid_numbers(self):
        is_valid, msg, result = verify_solution(
            "100 + 200",  # 200 not available
            300,
            [100, 25, 10, 5, 2, 1]
        )
        assert not is_valid
    
    def test_score_exact(self):
        score = score_solution("100 + 27", 127, [100, 27, 10, 5, 2, 1])
        assert score == 1.0
    
    def test_score_near_miss(self):
        score = score_solution("100 + 25", 127, [100, 25, 10, 5, 2, 1])
        assert 0.5 < score < 1.0


class TestGenerator:
    def test_generate_puzzle(self):
        puzzle = generate_puzzle(difficulty="medium")
        assert 101 <= puzzle.target <= 999
        assert len(puzzle.numbers) == 6
        assert puzzle.solution is not None
    
    def test_generate_easy_puzzle(self):
        puzzle = generate_puzzle(difficulty="easy")
        assert puzzle.difficulty == "easy"
        # Easy puzzles have only small numbers
        assert all(n <= 10 for n in puzzle.numbers)
    
    def test_solvable_guarantee(self):
        """All generated puzzles should be solvable."""
        for _ in range(10):
            puzzle = generate_puzzle(difficulty="medium", ensure_solvable=True)
            is_valid, _, _ = verify_solution(
                puzzle.solution, 
                puzzle.target, 
                puzzle.numbers
            )
            assert is_valid


class TestEnvironment:
    def test_load_environment(self):
        env = load_environment(num_examples=10)
        assert len(env.dataset) == 10
    
    def test_environment_structure(self):
        env = load_environment(num_examples=5)
        example = env.dataset[0]
        
        assert "prompt" in example
        assert "answer" in example
        assert "info" in example
        assert "target" in example["info"]
        assert "numbers" in example["info"]
    
    @pytest.mark.asyncio
    async def test_rubric_rewards(self):
        """Test that rubric produces expected rewards."""
        env = load_environment(num_examples=1)
        
        # Get a puzzle
        example = env.dataset[0]
        target = example["info"]["target"]
        numbers = example["info"]["numbers"]
        solution = example["info"]["solution"]
        
        # Simulate correct completion
        correct_completion = [
            {"role": "assistant", "content": f"<answer>{solution}</answer>"}
        ]
        
        # Compute reward
        reward = await env.rubric.evaluate(
            prompt=example["prompt"],
            completion=correct_completion,
            answer=example["answer"],
            info=example["info"],
        )
        
        assert reward > 0.9  # Should be close to 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 8.2 Run Tests

```bash
cd environments/countdown_numbers
pytest tests/ -v
```

### 8.3 Integration Test

```bash
# Full pipeline test
uv run vf-install countdown_numbers
uv run vf-eval countdown_numbers -m gpt-4.1-mini -n 5 -r 1

# Should see output like:
# Example 1: reward=0.95 (exact solve)
# Example 2: reward=0.72 (off by 3)
# ...
```

---

## Quick Start Checklist

```bash
# 1. Initialize environment
cd /path/to/verifiers
uv run vf-init countdown_numbers

# 2. Copy the implementation files:
#    - generator.py
#    - solver.py  
#    - countdown_numbers.py
#    - __init__.py
#    - pyproject.toml

# 3. Install and test
uv run vf-install countdown_numbers
python -c "from countdown_numbers import play_countdown; play_countdown()"

# 4. Run evaluation
uv run vf-eval countdown_numbers -m gpt-4.1-mini -n 10

# 5. Setup training
uv run vf-setup
# Edit configs/vf-rl/countdown_numbers.toml

# 6. Start training
uv run vf-rl @ configs/vf-rl/countdown_numbers.toml
```

---

## Potential Extensions

1. **Multi-turn version**: Model gets feedback after each guess (like Wordle)
2. **Time pressure**: Token budget simulates 30-second countdown
3. **Harder constraints**: Must use all numbers, or reach exact target
4. **Explanation grading**: Use LLM judge to score explanation quality
5. **Adversarial generation**: Generate puzzles that the model struggles with

---

## References

- [Verifiers Documentation](https://docs.primeintellect.ai/verifiers)
- [Prime-RL Training](https://github.com/PrimeIntellect-ai/prime-rl)
- [Countdown (TV Show)](https://en.wikipedia.org/wiki/Countdown_(game_show))
- [Numbers Game Solver Algorithms](https://www.datagenetics.com/blog/august32014/)