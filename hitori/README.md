# Hitori Puzzle RL Training Environment

A complete environment for training language models to solve Hitori puzzles using GRPO (Group Relative Policy Optimization).

## What is Hitori?

Hitori is a logic puzzle played on a grid of numbers. The goal is to shade certain cells while following three rules:

1. **Uniqueness**: Each number must appear only once per row and column among unshaded cells
2. **Non-Adjacency**: No two shaded cells may be adjacent horizontally or vertically
3. **Connectivity**: All unshaded cells must form a single connected region

## Quick Start

### 0. Download Base Model (Optional)

Download the model locally for faster loading:

```bash
# Download Qwen2.5-3B-Instruct to models/qwen2.5-3b-instruct
python download_model.py

# Or specify a different model/location
python download_model.py --model Qwen/Qwen2.5-3B-Instruct --output models/qwen2.5-3b-instruct
```

**Default model location**: `models/qwen2.5-3b-instruct`

### 1. Generate Dataset

```bash
# Generate default dataset (10k train, 200 per eval difficulty)
python dataset.py

# Custom generation
python dataset.py --train-examples 5000 --eval-per-difficulty 100 --output-dir data/hitoridata
```

### 2. Test Single Rollout

Before training, test what the model output looks like:

```bash
# Test with base model
python test_rollout.py --model-path Qwen/Qwen2.5-3B-Instruct --difficulty medium

# Test with local model
python test_rollout.py --model-path models/qwen2.5-3b-instruct --difficulty easy

# Test with 4-bit quantization (lower memory)
python test_rollout.py --model-path models/qwen2.5-3b-instruct --load-in-4bit
```

### 3. Train Model

```bash
# Single GPU (slow, for testing)
python train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo

# With Wandb logging
python train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo \
    --report_to wandb \
    --wandb_project hitori-grpo \
    --wandb_run_name my-experiment

# With HuggingFace Hub checkpoint upload
python train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo \
    --hub_model_id username/hitori-grpo-model \
    --hub_private true

# Multi-GPU with DeepSpeed
accelerate launch --num_processes 4 --config_file configs/deepspeed_zero3.yaml \
    train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo \
    --use_vllm true \
    --report_to wandb
```

### 4. Evaluate

```bash
python eval_hitori.py \
    --model-path outputs/hitori-grpo \
    --eval-data data/hitoridata \
    --output-dir eval_results
```

## File Structure

```
hitori/
├── solver.py              # Solution verification and solving algorithms
├── generator.py           # Puzzle generation with difficulty levels
├── dataset.py             # Dataset creation for train/eval splits
├── train_hitori_grpo.py   # GRPO training script (TRL-based)
├── eval_hitori.py         # Evaluation script
├── download_model.py      # Model download script
├── test_rollout.py        # Single rollout testing script
├── requirements.txt       # Dependencies
├── README.md              # This file
├── models/                # Downloaded models (created by download_model.py)
│   └── qwen2.5-3b-instruct/
├── data/                  # Generated datasets (created by dataset.py)
│   └── hitoridata/
└── outputs/               # Training outputs
    └── hitori-grpo/
```

## Puzzle Representation

### Grid Format

6x6 grid with column labels A-F and row labels 1-6:

```
  A B C D E F
1 2 1 3 4 5 2
2 4 2 1 3 6 5
3 1 3 4 2 2 6
4 3 4 2 1 5 1
5 5 6 5 6 1 3
6 6 5 6 5 4 2
```

### Solution Format

Cells to shade listed as coordinates: `A1, C3, D5, F2`

### Prompt Format (R1-Style)

```
System: You are a puzzle solver. You think carefully about the problem step by step...

User: Solve this 6x6 Hitori puzzle:
[grid]
Rules:
1. UNIQUENESS: Each number must appear only once per row/column (unshaded)
2. NON-ADJACENCY: No two shaded cells may be adjacent
3. CONNECTIVITY: All unshaded cells must form a single connected region

Show your reasoning in <think> </think> tags.
Return the cells to shade in <answer> </answer> tags.
Example: <answer>A1, B3, C5, F2</answer>
```

## Reward Design

The training uses two binary reward functions following the GRPO framework. Both rewards are **binary** (1.0 or 0.0) - no partial credit.

### 1. Format Reward (`format_reward_func`)

Validates that the model's output follows the expected structure:

```
<think>
[reasoning steps]
</think>
<answer>
[comma-separated coordinates]
</answer>
```

**Implementation:**
```python
def format_reward_func(completions, target, **kwargs):
    """
    Returns:
        1.0 if output matches: <think>...</think><answer>...</answer>
        0.0 otherwise
    """
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>\s*<answer>([\s\S]*?)</answer>"
    # Check if completion matches the expected format
```

**Rationale**: The format reward encourages the model to:
- Think before answering (chain-of-thought)
- Provide structured, parseable output
- Follow the R1-style reasoning pattern

### 2. Solution Reward (`solution_reward_func`)

Verifies that the proposed solution satisfies all three Hitori constraints:

```python
def solution_reward_func(completions, target, grid, solution, **kwargs):
    """
    Returns:
        1.0 if solution satisfies ALL constraints:
            - Uniqueness: each number appears once per row/column (unshaded)
            - Non-adjacency: no two shaded cells are adjacent
            - Connectivity: all unshaded cells form connected region
        0.0 if ANY constraint is violated or answer cannot be parsed

    Extracts the LAST <answer> tag (final answer only).
    """
```

**Constraint Verification:**

| Constraint | Check Method | Complexity |
|------------|--------------|------------|
| Uniqueness | Count numbers per row/col among unshaded | O(n^2) |
| Adjacency | Check 4-neighbors of each shaded cell | O(shaded) |
| Connectivity | BFS from any unshaded cell | O(n^2) |

**Why Binary Rewards?**

- **Simplicity**: Clear learning signal - correct or incorrect
- **Avoiding reward hacking**: Partial credit could encourage gaming specific constraints
- **Matches evaluation**: The model either solves puzzles or doesn't
- **Following GRPO best practices**: The original DeepSeekMath paper used binary rewards

### Reward Combination

During GRPO training, both rewards contribute equally:

```python
trainer = GRPOTrainer(
    reward_funcs=[format_reward_func, solution_reward_func],
    ...
)
```

The model must achieve both format correctness AND solution correctness to receive maximum reward.

## Difficulty Levels

| Level | Shaded Cells | Description |
|-------|--------------|-------------|
| Easy | 4-6 | Few cells to shade, obvious deductions |
| Medium | 6-9 | Moderate complexity, some reasoning required |
| Hard | 11-14 | Many cells to shade, complex constraints |

### Training Distribution

- **Training set**: 100% medium difficulty (10,000 total)
- **Evaluation sets**: 200 puzzles each for easy, medium, hard, and mixed

## Generation Algorithm

### Step 1: Generate Valid Shading Pattern

```python
def generate_valid_shading(size, num_shaded):
    """
    Randomly place shaded cells while respecting:
    - No adjacent shaded cells
    - Remaining cells stay connected
    """
```

### Step 2: Assign Numbers

```python
def assign_numbers(size, shaded):
    """
    1. Assign unique numbers to unshaded cells (Latin square subset)
    2. Assign duplicate numbers to shaded cells (create puzzle constraints)
    """
```

### Step 3: Verify Unique Solution

```python
def has_unique_solution(grid):
    """
    Use constraint propagation + backtracking to:
    1. Find all valid solutions
    2. Return True only if exactly one solution exists
    """
```

## Solver Implementation

The solver uses constraint propagation with backtracking:

```python
class HitoriSolver:
    def __init__(self, grid):
        self.grid = grid
        self.state = [[UNKNOWN] * size for _ in range(size)]

    def propagate(self):
        """Apply logical deductions:
        - If a number appears once in row/col, it must be unshaded
        - If shading a cell would disconnect unshaded cells, don't shade it
        - If shading a cell would create adjacent shaded cells, don't shade it
        """

    def solve(self):
        """
        1. Propagate constraints
        2. If stuck, pick cell with fewest possibilities
        3. Try shading/not shading with backtracking
        4. Return solution or None
        """
```

## Training Configuration

Recommended hyperparameters (based on mini-deepseek-r1-aha-grpo):

```yaml
# Model
model_name_or_path: Qwen/Qwen2.5-3B-Instruct

# GRPO
learning_rate: 5e-7
beta: 0.001  # KL coefficient
num_generations: 8  # Samples per prompt
max_completion_length: 1024

# Training
num_train_epochs: 3
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Generation
temperature: 0.7
top_p: 0.9
```

## Evaluation Metrics

The evaluation script reports:

- **Format Accuracy**: % of outputs with correct `<think>...<answer>` structure
- **Solution Accuracy**: % of puzzles solved correctly (all constraints satisfied)
- **Constraint Breakdown**:
  - Uniqueness satisfaction rate
  - Adjacency satisfaction rate
  - Connectivity satisfaction rate
- **Per-difficulty accuracy**: Breakdown by easy/medium/hard

## Wandb Integration

Training metrics are automatically logged to Weights & Biases when enabled:

```bash
# Enable wandb logging
python train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo \
    --report_to wandb \
    --wandb_project hitori-grpo \
    --wandb_run_name experiment-v1 \
    --wandb_tags "baseline,medium-difficulty"
```

**Logged metrics:**
- Training loss and rewards
- Format reward rate
- Solution reward rate
- Learning rate schedule
- GPU memory usage

**Environment variables:**
```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=hitori-grpo  # Optional, can also use --wandb_project
```

## HuggingFace Hub Integration

Automatically upload checkpoints to HuggingFace Hub:

```bash
# Upload to HF Hub
python train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo \
    --hub_model_id username/hitori-grpo-model \
    --hub_private true

# Combined with wandb
python train_hitori_grpo.py \
    --model_name_or_path models/qwen2.5-3b-instruct \
    --output_dir outputs/hitori-grpo \
    --report_to wandb \
    --wandb_project hitori-grpo \
    --hub_model_id username/hitori-grpo-model
```

**Authentication:**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN=your_token
```

## Requirements

- Python 3.10+
- PyTorch 2.5+
- Transformers 4.48+
- TRL 0.14+
- vLLM 0.7+ (for fast generation)
- DeepSpeed 0.15+ (for distributed training)
- Wandb 0.16+ (for experiment tracking)

Install dependencies:

```bash
pip install -r requirements.txt
```

## References

- [GRPO Paper (DeepSeekMath)](https://arxiv.org/abs/2402.03300)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Hitori Puzzle Rules](https://www.nikoli.co.jp/en/puzzles/hitori/)
- [mini-deepseek-r1-aha-grpo notebook](https://github.com/philschmid/deep-learning-pytorch-huggingface)