"""
Hitori Puzzle Solver and Verification Module.

Provides functions to verify Hitori solutions and solve puzzles using
constraint propagation with backtracking.

Hitori Rules:
1. UNIQUENESS: Each number appears at most once per row/column (unshaded cells only)
2. NON-ADJACENCY: No two shaded cells may be orthogonally adjacent
3. CONNECTIVITY: All unshaded cells must form a single connected region
"""

from collections import deque
from typing import List, Set, Tuple, Optional, Dict
from copy import deepcopy

GRID_SIZE = 6
COLUMNS = "ABCDEF"


def coord_to_notation(row: int, col: int) -> str:
    """Convert (row, col) tuple to notation like 'A1', 'B3'."""
    return f"{COLUMNS[col]}{row + 1}"


def notation_to_coord(notation: str) -> Tuple[int, int]:
    """Convert notation like 'A1', 'B3' to (row, col) tuple."""
    notation = notation.strip().upper()
    col = COLUMNS.index(notation[0])
    row = int(notation[1]) - 1
    return (row, col)


def parse_coordinates(text: str) -> Set[Tuple[int, int]]:
    """
    Parse coordinate notation from text.

    Accepts formats like:
    - "A1, B2, C3"
    - "A1 B2 C3"
    - "A1,B2,C3"

    Returns set of (row, col) tuples.
    """
    import re

    # Find all valid coordinate patterns (letter A-F followed by digit 1-6)
    pattern = r'[A-Fa-f][1-6]'
    matches = re.findall(pattern, text)

    coords = set()
    for match in matches:
        try:
            coords.add(notation_to_coord(match))
        except (ValueError, IndexError):
            continue

    return coords


def format_solution(shaded: Set[Tuple[int, int]]) -> str:
    """Format solution as coordinate string."""
    if not shaded:
        return ""
    sorted_coords = sorted(shaded, key=lambda x: (x[0], x[1]))
    return ", ".join(coord_to_notation(r, c) for r, c in sorted_coords)


# =============================================================================
# Constraint Verification Functions
# =============================================================================

def check_uniqueness(grid: List[List[int]], shaded: Set[Tuple[int, int]]) -> Tuple[bool, Dict]:
    """
    Check uniqueness constraint: each number appears at most once per row/column
    among unshaded cells.

    Returns:
        (is_valid, details) where details contains row/column violation info
    """
    size = len(grid)
    violations = {"rows": [], "cols": []}

    # Check rows
    for row in range(size):
        unshaded_values = []
        for col in range(size):
            if (row, col) not in shaded:
                unshaded_values.append(grid[row][col])

        if len(unshaded_values) != len(set(unshaded_values)):
            violations["rows"].append(row)

    # Check columns
    for col in range(size):
        unshaded_values = []
        for row in range(size):
            if (row, col) not in shaded:
                unshaded_values.append(grid[row][col])

        if len(unshaded_values) != len(set(unshaded_values)):
            violations["cols"].append(col)

    is_valid = len(violations["rows"]) == 0 and len(violations["cols"]) == 0
    return is_valid, violations


def check_adjacency(shaded: Set[Tuple[int, int]], size: int = GRID_SIZE) -> Tuple[bool, List]:
    """
    Check non-adjacency constraint: no two shaded cells are orthogonally adjacent.

    Returns:
        (is_valid, list of adjacent pairs)
    """
    adjacent_pairs = []

    for (r, c) in shaded:
        # Check right neighbor
        if (r, c + 1) in shaded:
            adjacent_pairs.append(((r, c), (r, c + 1)))
        # Check bottom neighbor
        if (r + 1, c) in shaded:
            adjacent_pairs.append(((r, c), (r + 1, c)))

    return len(adjacent_pairs) == 0, adjacent_pairs


def check_connectivity(shaded: Set[Tuple[int, int]], size: int = GRID_SIZE) -> Tuple[bool, Dict]:
    """
    Check connectivity constraint: all unshaded cells form a single connected region.
    Uses BFS from any unshaded cell.

    Returns:
        (is_valid, details with total_unshaded and connected_count)
    """
    # Get all unshaded cells
    unshaded = {(r, c) for r in range(size) for c in range(size) if (r, c) not in shaded}

    if not unshaded:
        # Edge case: all cells shaded (invalid puzzle state)
        return False, {"total_unshaded": 0, "connected_count": 0}

    # BFS from first unshaded cell
    start = next(iter(unshaded))
    visited = set()
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # Check all 4 neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in unshaded and (nr, nc) not in visited:
                queue.append((nr, nc))

    is_connected = len(visited) == len(unshaded)
    return is_connected, {"total_unshaded": len(unshaded), "connected_count": len(visited)}


def verify_solution(grid: List[List[int]], shaded: Set[Tuple[int, int]]) -> Tuple[bool, Dict]:
    """
    Verify a complete Hitori solution against all three rules.

    Args:
        grid: 2D list of numbers (the puzzle)
        shaded: Set of (row, col) tuples for shaded cells

    Returns:
        (is_valid, detailed_results) where detailed_results contains:
        - uniqueness: (bool, violations)
        - adjacency: (bool, pairs)
        - connectivity: (bool, details)
        - all_valid: bool
    """
    size = len(grid)

    uniqueness_valid, uniqueness_details = check_uniqueness(grid, shaded)
    adjacency_valid, adjacency_details = check_adjacency(shaded, size)
    connectivity_valid, connectivity_details = check_connectivity(shaded, size)

    all_valid = uniqueness_valid and adjacency_valid and connectivity_valid

    return all_valid, {
        "uniqueness": (uniqueness_valid, uniqueness_details),
        "adjacency": (adjacency_valid, adjacency_details),
        "connectivity": (connectivity_valid, connectivity_details),
        "all_valid": all_valid
    }


# =============================================================================
# Hitori Solver (Constraint Propagation + Backtracking)
# =============================================================================

class HitoriSolver:
    """
    Solves Hitori puzzles using constraint propagation with backtracking.
    Can also count solutions to verify uniqueness.
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.size = len(grid)
        # Cell states: None = unknown, True = shaded (black), False = unshaded (white)
        self.state = [[None for _ in range(self.size)] for _ in range(self.size)]

    def copy(self) -> 'HitoriSolver':
        """Create a deep copy of the solver state."""
        solver = HitoriSolver(self.grid)
        solver.state = deepcopy(self.state)
        return solver

    def get_shaded(self) -> Set[Tuple[int, int]]:
        """Get set of shaded cells."""
        return {(r, c) for r in range(self.size) for c in range(self.size)
                if self.state[r][c] is True}

    def get_unshaded(self) -> Set[Tuple[int, int]]:
        """Get set of unshaded cells."""
        return {(r, c) for r in range(self.size) for c in range(self.size)
                if self.state[r][c] is False}

    def get_unknown(self) -> Set[Tuple[int, int]]:
        """Get set of unknown cells."""
        return {(r, c) for r in range(self.size) for c in range(self.size)
                if self.state[r][c] is None}

    def set_shaded(self, row: int, col: int) -> bool:
        """
        Mark cell as shaded. Returns False if this creates a contradiction.
        """
        if self.state[row][col] is False:
            return False  # Contradiction: already marked unshaded
        if self.state[row][col] is True:
            return True  # Already shaded

        self.state[row][col] = True

        # Check adjacency: neighbors must be unshaded
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self.state[nr][nc] is True:
                    return False  # Adjacent shaded cells
                if self.state[nr][nc] is None:
                    if not self.set_unshaded(nr, nc):
                        return False

        return True

    def set_unshaded(self, row: int, col: int) -> bool:
        """
        Mark cell as unshaded. Returns False if this creates a contradiction.
        """
        if self.state[row][col] is True:
            return False  # Contradiction: already marked shaded
        if self.state[row][col] is False:
            return True  # Already unshaded

        self.state[row][col] = False
        value = self.grid[row][col]

        # All other cells with same value in row/col must be shaded
        # Check row
        for c in range(self.size):
            if c != col and self.grid[row][c] == value:
                if self.state[row][c] is False:
                    return False  # Duplicate unshaded value in row
                if self.state[row][c] is None:
                    if not self.set_shaded(row, c):
                        return False

        # Check column
        for r in range(self.size):
            if r != row and self.grid[r][col] == value:
                if self.state[r][col] is False:
                    return False  # Duplicate unshaded value in column
                if self.state[r][col] is None:
                    if not self.set_shaded(r, col):
                        return False

        return True

    def propagate(self) -> bool:
        """
        Apply constraint propagation rules until no more deductions can be made.
        Returns False if a contradiction is found.
        """
        changed = True
        while changed:
            changed = False

            for r in range(self.size):
                for c in range(self.size):
                    if self.state[r][c] is not None:
                        continue

                    # Rule: If shading this cell would disconnect unshaded cells
                    # or create adjacent shaded cells, it must be unshaded

                    # Check if shading would create adjacent shaded cells
                    would_have_adjacent = False
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.state[nr][nc] is True:
                                would_have_adjacent = True
                                break

                    if would_have_adjacent:
                        if not self.set_unshaded(r, c):
                            return False
                        changed = True
                        continue

                    # Rule: If this is the only cell with its value in row AND column
                    # (among unknowns + unshaded), it should be unshaded
                    value = self.grid[r][c]

                    # Count same value in row (excluding already shaded)
                    row_same = [c2 for c2 in range(self.size)
                                if self.grid[r][c2] == value and self.state[r][c2] is not True]
                    # Count same value in col (excluding already shaded)
                    col_same = [r2 for r2 in range(self.size)
                                if self.grid[r2][c] == value and self.state[r2][c] is not True]

                    # If only one option in row AND column, it must be unshaded
                    if len(row_same) == 1 and len(col_same) == 1:
                        if not self.set_unshaded(r, c):
                            return False
                        changed = True
                        continue

            # Check connectivity: if shading any unknown cell would disconnect,
            # that cell must be unshaded
            for r in range(self.size):
                for c in range(self.size):
                    if self.state[r][c] is None:
                        # Temporarily shade this cell
                        test_shaded = self.get_shaded() | {(r, c)}
                        is_connected, _ = check_connectivity(test_shaded, self.size)

                        if not is_connected:
                            if not self.set_unshaded(r, c):
                                return False
                            changed = True

        return True

    def is_complete(self) -> bool:
        """Check if all cells have been determined."""
        return all(self.state[r][c] is not None
                   for r in range(self.size) for c in range(self.size))

    def is_valid(self) -> bool:
        """Check if current state is valid (no contradictions)."""
        shaded = self.get_shaded()
        is_valid, _ = verify_solution(self.grid, shaded)
        return is_valid

    def solve(self, count_only: bool = False, max_solutions: int = 2) -> Optional[Set[Tuple[int, int]]]:
        """
        Solve the puzzle using constraint propagation + backtracking.

        Args:
            count_only: If True, just count solutions up to max_solutions
            max_solutions: Stop after finding this many solutions

        Returns:
            Set of shaded cells if solved, None if no solution
        """
        solutions = []
        self._solve_recursive(solutions, max_solutions)

        if count_only:
            return len(solutions)

        return solutions[0] if solutions else None

    def _solve_recursive(self, solutions: List, max_solutions: int):
        """Recursive backtracking solver."""
        if len(solutions) >= max_solutions:
            return

        # Apply constraint propagation
        if not self.propagate():
            return  # Contradiction

        # Check if complete
        if self.is_complete():
            shaded = self.get_shaded()
            is_valid, _ = verify_solution(self.grid, shaded)
            if is_valid:
                solutions.append(shaded.copy())
            return

        # Find an unknown cell to branch on
        unknown = self.get_unknown()
        if not unknown:
            return

        # Choose cell with most constraints (simple heuristic: most duplicates)
        def constraint_score(cell):
            r, c = cell
            value = self.grid[r][c]
            row_count = sum(1 for c2 in range(self.size) if self.grid[r][c2] == value)
            col_count = sum(1 for r2 in range(self.size) if self.grid[r2][c] == value)
            return row_count + col_count

        branch_cell = max(unknown, key=constraint_score)
        r, c = branch_cell

        # Try shading this cell
        solver_copy = self.copy()
        if solver_copy.set_shaded(r, c):
            solver_copy._solve_recursive(solutions, max_solutions)

        if len(solutions) >= max_solutions:
            return

        # Try not shading this cell
        solver_copy = self.copy()
        if solver_copy.set_unshaded(r, c):
            solver_copy._solve_recursive(solutions, max_solutions)


def solve_hitori(grid: List[List[int]]) -> Optional[Set[Tuple[int, int]]]:
    """
    Solve a Hitori puzzle.

    Args:
        grid: 2D list of numbers

    Returns:
        Set of (row, col) tuples for cells to shade, or None if unsolvable
    """
    solver = HitoriSolver(grid)
    return solver.solve()


def count_solutions(grid: List[List[int]], max_count: int = 2) -> int:
    """
    Count number of solutions for a Hitori puzzle.

    Args:
        grid: 2D list of numbers
        max_count: Stop counting after this many

    Returns:
        Number of solutions found (up to max_count)
    """
    solver = HitoriSolver(grid)
    return solver.solve(count_only=True, max_solutions=max_count)


def has_unique_solution(grid: List[List[int]]) -> bool:
    """Check if puzzle has exactly one solution."""
    return count_solutions(grid, max_count=2) == 1


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test puzzle (simple example)
    test_grid = [
        [1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 1],
        [3, 4, 5, 6, 1, 2],
        [4, 5, 6, 1, 2, 3],
        [5, 6, 1, 2, 3, 4],
        [6, 1, 2, 3, 4, 5],
    ]

    print("Test grid (Latin square - no shading needed):")
    for row in test_grid:
        print(row)

    # This grid has no duplicates, so solution is empty set
    solution = solve_hitori(test_grid)
    print(f"\nSolution: {solution}")

    # Verify
    if solution is not None:
        is_valid, details = verify_solution(test_grid, solution)
        print(f"Valid: {is_valid}")
        print(f"Details: {details}")

    # Test coordinate parsing
    print("\nCoordinate parsing tests:")
    test_input = "A1, B2, C3, F6"
    parsed = parse_coordinates(test_input)
    print(f"Input: {test_input}")
    print(f"Parsed: {parsed}")
    print(f"Formatted: {format_solution(parsed)}")
