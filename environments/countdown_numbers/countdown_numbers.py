"""
Countdown Numbers Environment for verifiers.

A mathematical puzzle game based on the UK TV show "Countdown" where players
combine given numbers using arithmetic operations to reach a target value.

Rules:
- Given: 6 numbers selected from large (25, 50, 75, 100) and small (1-10) numbers
- Target: A random 3-digit number (101-999)
- Goal: Combine numbers using +, -, *, / to reach the target exactly
- Constraints:
  - Each given number can only be used once
  - All intermediate results must be positive integers
  - Division must be exact (no remainders)
  - Not all numbers need to be used
"""
import random
import re
from dataclasses import dataclass
from itertools import permutations, product
from typing import Optional

from datasets import Dataset

import verifiers as vf


# =============================================================================
# Puzzle Generation
# =============================================================================

LARGE_NUMBERS = [25, 50, 75, 100]
SMALL_NUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@dataclass
class CountdownPuzzle:
    """Represents a Countdown Numbers puzzle."""

    target: int
    numbers: list[int]
    solution: Optional[str] = None
    difficulty: str = "medium"


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
            if token == "+":
                result = a + b
            elif token == "-":
                result = a - b
                if result <= 0:
                    return None
            elif token == "*":
                result = a * b
            elif token == "/":
                if b == 0 or a % b != 0:
                    return None
                result = a // b
            else:
                return None
            stack.append(result)

    return stack[0] if len(stack) == 1 else None


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


def solve_countdown(
    numbers: list[int], target: int, max_solutions: int = 1
) -> list[str]:
    """
    Find solutions using brute-force search over all combinations.
    Returns list of solution strings in infix notation.
    """
    solutions = []
    ops = ["+", "-", "*", "/"]

    # Try all subsets of numbers (2 to 6 numbers)
    for r in range(2, len(numbers) + 1):
        for perm in permutations(numbers, r):
            # Try all operator combinations
            for op_combo in product(ops, repeat=r - 1):
                # Build RPN: n1 n2 op1 n3 op2 ... (left-to-right evaluation)
                rpn = [perm[0], perm[1], op_combo[0]]
                for i in range(2, r):
                    rpn.extend([perm[i], op_combo[i - 1]])

                result = evaluate_rpn(rpn)
                if result == target:
                    infix = rpn_to_infix(rpn)
                    if infix not in solutions:
                        solutions.append(infix)
                        if len(solutions) >= max_solutions:
                            return solutions

    return solutions


def generate_puzzle_from_solution(difficulty: str = "medium") -> CountdownPuzzle:
    """
    Generate puzzle by working backwards from a valid solution.
    Guarantees solvability.
    """
    difficulty_config = {
        "easy": {"num_large": 0, "ops": ["+"], "steps": 2},
        "medium": {"num_large": 2, "ops": ["+", "-", "*"], "steps": 3},
        "hard": {"num_large": 4, "ops": ["+", "-", "*", "/"], "steps": 4},
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

        if op == "+":
            result = result + n
        elif op == "-":
            if result > n:
                result = result - n
            else:
                op = "+"
                result = result + n
        elif op == "*":
            result = result * n
        elif op == "/":
            if n != 0 and result % n == 0:
                result = result // n
            else:
                op = "+"
                result = result + n

        expression_parts.append(f"{op} {n}")

    solution = " ".join(expression_parts)

    return CountdownPuzzle(
        target=result, numbers=numbers, solution=solution, difficulty=difficulty
    )


def generate_puzzle(difficulty: str = "medium", ensure_solvable: bool = True) -> CountdownPuzzle:
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
                    difficulty=difficulty,
                )
        else:
            return CountdownPuzzle(
                target=target, numbers=numbers, difficulty=difficulty
            )

    # Fallback: generate from solution (guaranteed solvable)
    return generate_puzzle_from_solution(difficulty)


# =============================================================================
# Solution Verification
# =============================================================================


def parse_expression(expr: str) -> Optional[int]:
    """
    Safely evaluate a mathematical expression.
    Only allows +, -, *, /, parentheses, and integers.

    Returns the result or None if invalid.
    """
    expr = expr.strip()

    # Remove the "= result" part if present
    if "=" in expr:
        expr = expr.split("=")[0].strip()

    # Validate characters (only digits, operators, parens, spaces)
    if not re.match(r"^[\d\s\+\-\*\/\(\)]+$", expr):
        return None

    try:
        # Use Python's eval with restricted globals
        result = eval(expr, {"__builtins__": {}}, {})

        # Check it's a positive integer
        if isinstance(result, (int, float)):
            if result == int(result) and result > 0:
                return int(result)
        return None
    except Exception:
        return None


def extract_numbers_used(expr: str) -> list[int]:
    """Extract all numbers used in an expression."""
    if "=" in expr:
        expr = expr.split("=")[0]

    numbers = re.findall(r"\d+", expr)
    return [int(n) for n in numbers]


def verify_solution(
    expr: str, target: int, available_numbers: list[int]
) -> tuple[bool, str, Optional[int]]:
    """
    Verify a Countdown solution.

    Args:
        expr: The expression to verify (e.g., "(100 + 25) * 4")
        target: The target number to reach
        available_numbers: The numbers that were available

    Returns:
        (is_valid, message, result)
    """
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


def score_solution(expr: str, target: int, available_numbers: list[int]) -> float:
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
    is_valid, _, result = verify_solution(expr, target, available_numbers)

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


# =============================================================================
# Dataset Builder
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are playing the Countdown Numbers game. Your task is to combine the given numbers using arithmetic operations (+, -, *, /) to reach the target exactly.

Rules:
- Each number can only be used once
- All intermediate results must be positive integers
- Division must be exact (no remainders)
- You don't have to use all numbers

Think step-by-step, then provide your final answer in this format:
<answer>YOUR_EXPRESSION = TARGET</answer>

For example: <answer>(100 + 25) * 4 = 500</answer>"""


def get_dataset_builder(
    num_examples: int = 1000,
    difficulty: str = "mixed",
    seed: int = 42,
) -> vf.DatasetBuilder:
    """
    Returns a DatasetBuilder that lazily builds the Countdown Numbers dataset.

    Args:
        num_examples: Number of examples to generate
        difficulty: "easy", "medium", "hard", or "mixed"
        seed: Random seed for reproducibility
    """

    def build() -> Dataset:
        random.seed(seed)

        if difficulty == "mixed":
            difficulty_distribution = {"easy": 0.2, "medium": 0.6, "hard": 0.2}
        else:
            difficulty_distribution = {difficulty: 1.0}

        data = []
        for _ in range(num_examples):
            # Sample difficulty
            r = random.random()
            cumsum = 0
            selected_difficulty = "medium"
            for diff, prob in difficulty_distribution.items():
                cumsum += prob
                if r < cumsum:
                    selected_difficulty = diff
                    break

            puzzle = generate_puzzle(difficulty=selected_difficulty, ensure_solvable=True)

            # Format as chat message
            prompt_text = f"""Target: {puzzle.target}
Available numbers: {', '.join(map(str, puzzle.numbers))}

Combine the numbers using +, -, *, / to reach the target exactly. Each number can only be used once, and all intermediate results must be positive integers."""

            data.append(
                {
                    "prompt": [{"role": "user", "content": prompt_text}],
                    "answer": str(puzzle.target),
                    "info": {
                        "target": puzzle.target,
                        "numbers": puzzle.numbers,
                        "solution": puzzle.solution,
                        "difficulty": puzzle.difficulty,
                    },
                }
            )

        return Dataset.from_list(data)

    return build


# =============================================================================
# Reward Functions
# =============================================================================


def extract_answer_from_tags(text: str) -> Optional[str]:
    """Extract the expression from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def correctness_reward(completion, info, **kwargs) -> float:
    """
    Main reward: how close is the answer to the target?
    1.0 for exact match, scaled down for near misses.
    """
    last_msg = completion[-1]["content"] if completion else ""

    parsed = extract_answer_from_tags(last_msg)
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

    parsed = extract_answer_from_tags(last_msg)
    if parsed is None:
        return 0.0

    # Check it looks like a math expression with operators
    if not re.search(r"\d+\s*[\+\-\*\/]", parsed):
        return 0.5

    return 1.0


def efficiency_reward(completion, info, **kwargs) -> float:
    """
    Bonus for elegant solutions (fewer numbers used).
    Only applies if the answer is correct.
    """
    last_msg = completion[-1]["content"] if completion else ""
    parsed = extract_answer_from_tags(last_msg)

    if parsed is None:
        return 0.0

    target = info["target"]
    numbers = info["numbers"]

    is_valid, _, _ = verify_solution(parsed, target, numbers)

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
    calc_pattern = r"\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+"
    calculations = re.findall(calc_pattern, last_msg)

    # Also count lines that look like reasoning
    lines = last_msg.split("\n")
    reasoning_lines = sum(
        1
        for line in lines
        if "=" in line or any(op in line for op in ["+", "-", "*", "/"])
    )

    # Reward for showing work
    if len(calculations) >= 2 or reasoning_lines >= 3:
        return 1.0
    elif len(calculations) >= 1 or reasoning_lines >= 2:
        return 0.5
    return 0.0


# =============================================================================
# Environment Loader
# =============================================================================


def load_environment(
    num_train_examples: int = 1000,
    num_eval_examples: int = 100,
    difficulty: str = "mixed",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    """
    Load the Countdown Numbers environment.

    Args:
        num_train_examples: Number of training examples (-1 for default)
        num_eval_examples: Number of evaluation examples
        difficulty: "easy", "medium", "hard", or "mixed"
        system_prompt: System prompt for the model
        seed: Random seed for dataset generation

    Returns:
        Configured verifiers Environment
    """
    # Create dataset builders for lazy loading
    train_builder = get_dataset_builder(
        num_examples=num_train_examples if num_train_examples > 0 else 1000,
        difficulty=difficulty,
        seed=seed,
    )

    eval_builder = get_dataset_builder(
        num_examples=num_eval_examples if num_eval_examples > 0 else 100,
        difficulty=difficulty,
        seed=seed + 1000,  # Different seed for eval
    )

    # Create parser for XML answer tags
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")

    # Create rubric with weighted rewards
    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(correctness_reward, weight=1.0)
    rubric.add_reward_func(format_reward, weight=0.2)
    rubric.add_reward_func(efficiency_reward, weight=0.1)
    rubric.add_reward_func(reasoning_reward, weight=0.1)

    # Create environment
    env = vf.SingleTurnEnv(
        dataset=train_builder,
        eval_dataset=eval_builder,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

    return env
