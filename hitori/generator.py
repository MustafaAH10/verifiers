"""
Hitori Puzzle Generator.

Generates valid 6x6 Hitori puzzles with guaranteed unique solutions.
Supports three difficulty levels: easy, medium, hard.

Algorithm:
1. Generate valid shading pattern (respecting non-adjacency and connectivity)
2. Assign numbers to create the puzzle (duplicates for shaded cells)
3. Verify unique solution via constraint propagation solver
4. Retry if not uniquely solvable
"""

import random
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional
from collections import deque

from solver import (
    GRID_SIZE, COLUMNS,
    check_connectivity, check_adjacency,
    verify_solution, has_unique_solution, solve_hitori,
    format_solution, coord_to_notation
)


@dataclass
class HitoriPuzzle:
    """Represents a generated Hitori puzzle."""
    grid: List[List[int]]
    solution: Set[Tuple[int, int]]
    difficulty: str

    def format_grid(self) -> str:
        """Format grid for display with column headers."""
        lines = []
        # Header row
        header = "  " + " ".join(COLUMNS[:GRID_SIZE])
        lines.append(header)
        # Data rows
        for i, row in enumerate(self.grid):
            row_str = f"{i+1} " + " ".join(str(x) for x in row)
            lines.append(row_str)
        return "\n".join(lines)

    def format_solution(self) -> str:
        """Format solution as coordinate string."""
        return format_solution(self.solution)


# =============================================================================
# Difficulty Configuration
# =============================================================================

DIFFICULTY_CONFIG = {
    "easy": {
        "min_shaded": 4,
        "max_shaded": 6,
        "description": "Few cells to shade, obvious deductions"
    },
    "medium": {
        "min_shaded": 6,
        "max_shaded": 9,
        "description": "Moderate complexity, some reasoning required"
    },
    "hard": {
        "min_shaded": 11,
        "max_shaded": 14,
        "description": "Many cells to shade, complex constraints"
    }
}


# =============================================================================
# Shading Generation
# =============================================================================

def generate_valid_shading(
    size: int,
    num_shaded: int,
    max_attempts: int = 1000
) -> Optional[Set[Tuple[int, int]]]:
    """
    Generate a valid shading pattern that respects:
    - Non-adjacency: no two shaded cells are orthogonally adjacent
    - Connectivity: all unshaded cells form a single connected region

    Args:
        size: Grid size (e.g., 6 for 6x6)
        num_shaded: Target number of shaded cells
        max_attempts: Maximum attempts before giving up

    Returns:
        Set of (row, col) tuples for shaded cells, or None if failed
    """
    for _ in range(max_attempts):
        shaded = set()
        all_cells = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(all_cells)

        for (r, c) in all_cells:
            if len(shaded) >= num_shaded:
                break

            # Check if we can shade this cell
            # Rule 1: No adjacent shaded cells
            has_adjacent = any(
                (r + dr, c + dc) in shaded
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            )
            if has_adjacent:
                continue

            # Rule 2: Shading this cell shouldn't disconnect unshaded cells
            test_shaded = shaded | {(r, c)}
            is_connected, _ = check_connectivity(test_shaded, size)
            if not is_connected:
                continue

            shaded.add((r, c))

        # Check if we got enough shaded cells
        if len(shaded) >= num_shaded:
            return shaded

    return None


# =============================================================================
# Number Assignment
# =============================================================================

def assign_numbers(
    size: int,
    shaded: Set[Tuple[int, int]],
    max_attempts: int = 100
) -> Optional[List[List[int]]]:
    """
    Assign numbers to create a valid Hitori puzzle from a shading pattern.

    Strategy:
    1. First, assign numbers to unshaded cells ensuring uniqueness in rows/cols
    2. Then, assign numbers to shaded cells that create duplicates

    Args:
        size: Grid size
        shaded: Set of cells that should be shaded in the solution

    Returns:
        2D grid of numbers, or None if failed
    """
    for attempt in range(max_attempts):
        grid = [[0 for _ in range(size)] for _ in range(size)]
        unshaded = [(r, c) for r in range(size) for c in range(size) if (r, c) not in shaded]

        # Step 1: Assign numbers to unshaded cells (like a Latin square subset)
        success = True
        for (r, c) in unshaded:
            # Find valid numbers for this cell
            used_in_row = {grid[r][c2] for c2 in range(size) if (r, c2) not in shaded and grid[r][c2] != 0}
            used_in_col = {grid[r2][c] for r2 in range(size) if (r2, c) not in shaded and grid[r2][c] != 0}
            used = used_in_row | used_in_col

            available = [n for n in range(1, size + 1) if n not in used]
            if not available:
                success = False
                break

            grid[r][c] = random.choice(available)

        if not success:
            continue

        # Step 2: Assign numbers to shaded cells
        # Each shaded cell should duplicate a number in its row OR column
        for (r, c) in shaded:
            # Get unshaded numbers in the same row and column
            row_numbers = [grid[r][c2] for c2 in range(size) if (r, c2) not in shaded]
            col_numbers = [grid[r2][c] for r2 in range(size) if (r2, c) not in shaded]

            # Prefer numbers that appear in both (stronger constraint)
            common = set(row_numbers) & set(col_numbers)
            if common:
                grid[r][c] = random.choice(list(common))
            elif row_numbers:
                grid[r][c] = random.choice(row_numbers)
            elif col_numbers:
                grid[r][c] = random.choice(col_numbers)
            else:
                # Fallback: any valid number
                grid[r][c] = random.randint(1, size)

        # Verify this creates a valid puzzle with the intended solution
        solved = solve_hitori(grid)
        if solved is not None and solved == shaded:
            return grid

    return None


def generate_puzzle_with_shading(
    size: int,
    shaded: Set[Tuple[int, int]],
    max_attempts: int = 50
) -> Optional[List[List[int]]]:
    """
    Generate a puzzle that has exactly the given shading as its unique solution.
    Uses a more robust approach by trying multiple number assignments.
    """
    for _ in range(max_attempts):
        grid = assign_numbers(size, shaded, max_attempts=20)
        if grid is not None:
            # Double-check unique solution
            if has_unique_solution(grid):
                return grid
    return None


# =============================================================================
# Main Generation Function
# =============================================================================

def generate_puzzle(
    difficulty: str = "medium",
    size: int = GRID_SIZE,
    max_attempts: int = 100
) -> Optional[HitoriPuzzle]:
    """
    Generate a complete Hitori puzzle.

    Args:
        difficulty: "easy", "medium", or "hard"
        size: Grid size (default 6)
        max_attempts: Maximum generation attempts

    Returns:
        HitoriPuzzle object or None if generation failed
    """
    config = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["medium"])
    min_shaded = config["min_shaded"]
    max_shaded = config["max_shaded"]

    for attempt in range(max_attempts):
        # Choose target number of shaded cells
        num_shaded = random.randint(min_shaded, max_shaded)

        # Generate valid shading pattern
        shaded = generate_valid_shading(size, num_shaded)
        if shaded is None:
            continue

        # Assign numbers to create puzzle
        grid = generate_puzzle_with_shading(size, shaded)
        if grid is None:
            continue

        # Final verification
        is_valid, _ = verify_solution(grid, shaded)
        if is_valid and has_unique_solution(grid):
            return HitoriPuzzle(
                grid=grid,
                solution=shaded,
                difficulty=difficulty
            )

    return None


def generate_puzzles(
    num_puzzles: int,
    difficulty: str = "mixed",
    seed: int = 42,
    verbose: bool = False
) -> List[HitoriPuzzle]:
    """
    Generate multiple Hitori puzzles.

    Args:
        num_puzzles: Number of puzzles to generate
        difficulty: "easy", "medium", "hard", or "mixed"
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        List of HitoriPuzzle objects
    """
    random.seed(seed)
    puzzles = []

    if difficulty == "mixed":
        distribution = {"medium": 1.0}  # Training uses only medium difficulty
    else:
        distribution = {difficulty: 1.0}

    difficulties = list(distribution.keys())
    weights = list(distribution.values())

    for i in range(num_puzzles):
        if verbose and (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_puzzles} puzzles...")

        # Sample difficulty
        diff = random.choices(difficulties, weights=weights)[0]

        # Generate puzzle
        puzzle = None
        attempts = 0
        while puzzle is None and attempts < 10:
            puzzle = generate_puzzle(difficulty=diff)
            attempts += 1

        if puzzle is not None:
            puzzles.append(puzzle)
        elif verbose:
            print(f"Warning: Failed to generate puzzle {i + 1}")

    return puzzles


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_prompt(puzzle: HitoriPuzzle) -> str:
    """Format a puzzle as a prompt for the LLM."""
    prompt = f"""Solve this 6x6 Hitori puzzle:

{puzzle.format_grid()}

Rules:
- Shade cells so each number appears only once per row and column (among unshaded cells)
- No two shaded cells may be adjacent horizontally or vertically
- All unshaded cells must form a single connected region

Show your reasoning in <think></think> tags.
Return the cells to shade in <answer></answer> tags using coordinate notation (e.g., A1, B3, F6)."""

    return prompt


def format_r1_prompt(puzzle: HitoriPuzzle, tokenizer=None) -> str:
    """
    Format puzzle as R1-style prompt with system/user/assistant roles.

    Args:
        puzzle: HitoriPuzzle to format
        tokenizer: Optional tokenizer for chat template formatting

    Returns:
        Formatted prompt string
    """
    system_content = "You are a puzzle solver. You think carefully about the problem step by step in your mind before providing the answer."

    user_content = f"""Solve this 6x6 Hitori puzzle:

{puzzle.format_grid()}

Rules:
1. UNIQUENESS: Each number must appear only once per row and column (among unshaded/white cells)
2. NON-ADJACENCY: No two shaded/black cells may be adjacent horizontally or vertically
3. CONNECTIVITY: All unshaded/white cells must form a single connected region

Instructions:
- Think through the problem inside <think> </think> tags
- After </think>, provide your FINAL answer with ALL cells to shade in a SINGLE <answer> </answer> tag
- Do NOT use multiple answer tags - only ONE at the very end with the complete solution

Example format:
<think>reasoning here...</think>
<answer>A1, B3, C5, F2</answer>"""

    assistant_prefix = "Let me solve this step by step.\n<think>"

    if tokenizer is not None:
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_prefix}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    else:
        # Simple format without tokenizer
        return f"System: {system_content}\n\nUser: {user_content}\n\nAssistant: {assistant_prefix}"


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Hitori puzzle generation...")
    print("=" * 50)

    for diff in ["easy", "medium", "hard"]:
        print(f"\nGenerating {diff} puzzle...")
        puzzle = generate_puzzle(difficulty=diff)

        if puzzle:
            print(f"\nPuzzle ({diff}):")
            print(puzzle.format_grid())
            print(f"\nSolution: {puzzle.format_solution()}")
            print(f"Shaded cells: {len(puzzle.solution)}")

            # Verify
            is_valid, details = verify_solution(puzzle.grid, puzzle.solution)
            print(f"Valid: {is_valid}")
        else:
            print(f"Failed to generate {diff} puzzle")

    # Test batch generation
    print("\n" + "=" * 50)
    print("Testing batch generation (10 mixed puzzles)...")
    puzzles = generate_puzzles(10, difficulty="mixed", seed=42, verbose=True)
    print(f"Generated {len(puzzles)} puzzles")

    # Show difficulty distribution
    from collections import Counter
    difficulties = Counter(p.difficulty for p in puzzles)
    print(f"Difficulty distribution: {dict(difficulties)}")
