# Countdown Numbers Environment

A mathematical puzzle environment based on the UK TV show "Countdown" where players combine given numbers using arithmetic operations to reach a target value.

## Game Rules

- **Given:** 6 numbers selected from:
  - Large numbers: `{25, 50, 75, 100}`
  - Small numbers: `{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}` (each can appear twice)
- **Target:** A random 3-digit number (101-999)
- **Goal:** Combine the numbers using `+`, `-`, `*`, `/` to reach the target exactly
- **Constraints:**
  - Each given number can only be used once
  - All intermediate results must be positive integers
  - Division must be exact (no remainders)
  - Not all numbers need to be used

## Example

```
Target: 127
Numbers: 100, 25, 10, 5, 2, 1

Solution: 100 + 25 + 2 = 127
```

## Usage

```bash
# Install the environment
prime env install countdown_numbers

# Run evaluation
prime eval run countdown_numbers -m gpt-4.1-mini -n 20
```

## Configuration

The `load_environment` function accepts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_examples` | 1000 | Number of training examples |
| `num_eval_examples` | 100 | Number of evaluation examples |
| `difficulty` | "mixed" | "easy", "medium", "hard", or "mixed" |
| `seed` | 42 | Random seed for reproducibility |

### Difficulty Levels

- **Easy:** Only small numbers (1-10), targets 101-200, simple operations
- **Medium:** 2 large + 4 small numbers, targets 101-999, all operations
- **Hard:** 4 large + 2 small numbers, targets 500-999, complex solutions

## Reward Functions

| Function | Weight | Description |
|----------|--------|-------------|
| `correctness_reward` | 1.0 | Core reward - exact match = 1.0, partial credit for near misses |
| `format_reward` | 0.2 | Proper use of `<answer>` tags |
| `efficiency_reward` | 0.1 | Bonus for using fewer numbers |
| `reasoning_reward` | 0.1 | Bonus for showing step-by-step work |

## Expected Output Format

Models should respond with their solution in XML tags:

```
<answer>(100 + 25) * 4 = 500</answer>
```

## Why Countdown Numbers is Good for RL

- **Deterministic verification:** Arithmetic is easily checkable
- **Rich reward shaping:** Distance to target enables partial credit
- **Variable difficulty:** Easy (addition) to hard (complex expressions)
- **Procedurally generatable:** Infinite training data
- **Multi-step reasoning:** Requires search and planning
