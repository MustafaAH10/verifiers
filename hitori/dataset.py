"""
Hitori Dataset Creation Module.

Creates training and evaluation datasets for GRPO-based RL training.
Follows the format expected by TRL's GRPOTrainer.
"""

import os
import json
import random
from typing import List, Dict, Optional
from dataclasses import asdict

from datasets import Dataset, DatasetDict

from generator import generate_puzzles, format_r1_prompt, HitoriPuzzle
from solver import format_solution


def puzzle_to_dict(puzzle: HitoriPuzzle, tokenizer=None) -> Dict:
    """
    Convert a HitoriPuzzle to a dictionary format for the dataset.

    Args:
        puzzle: HitoriPuzzle object
        tokenizer: Optional tokenizer for prompt formatting

    Returns:
        Dictionary with prompt, target, grid, solution, difficulty
    """
    return {
        "prompt": format_r1_prompt(puzzle, tokenizer),
        "target": format_solution(puzzle.solution),
        "grid": puzzle.grid,
        "solution": [list(coord) for coord in puzzle.solution],  # Convert tuples to lists for JSON
        "difficulty": puzzle.difficulty,
        "nums": None  # Placeholder for compatibility with Countdown reward functions
    }


def create_training_dataset(
    num_examples: int = 10000,
    difficulty: str = "mixed",
    seed: int = 42,
    tokenizer=None,
    verbose: bool = True
) -> Dataset:
    """
    Create training dataset.

    Args:
        num_examples: Number of training examples
        difficulty: "easy", "medium", "hard", or "mixed"
        seed: Random seed for reproducibility
        tokenizer: Optional tokenizer for prompt formatting
        verbose: Print progress

    Returns:
        HuggingFace Dataset
    """
    if verbose:
        print(f"Generating {num_examples} training puzzles ({difficulty} difficulty)...")

    puzzles = generate_puzzles(
        num_puzzles=num_examples,
        difficulty=difficulty,
        seed=seed,
        verbose=verbose
    )

    if verbose:
        print(f"Successfully generated {len(puzzles)} puzzles")

    data = [puzzle_to_dict(p, tokenizer) for p in puzzles]
    return Dataset.from_list(data)


def create_eval_datasets(
    num_per_difficulty: int = 200,
    base_seed: int = 1000,
    tokenizer=None,
    verbose: bool = True
) -> Dict[str, Dataset]:
    """
    Create evaluation datasets for each difficulty level.

    Args:
        num_per_difficulty: Number of puzzles per difficulty
        base_seed: Base random seed (incremented for each difficulty)
        tokenizer: Optional tokenizer for prompt formatting
        verbose: Print progress

    Returns:
        Dictionary with keys "easy", "medium", "hard", "mixed"
    """
    eval_datasets = {}

    for i, diff in enumerate(["easy", "medium", "hard"]):
        seed = base_seed + i * 1000
        if verbose:
            print(f"\nGenerating {num_per_difficulty} {diff} eval puzzles (seed={seed})...")

        puzzles = generate_puzzles(
            num_puzzles=num_per_difficulty,
            difficulty=diff,
            seed=seed,
            verbose=verbose
        )

        data = [puzzle_to_dict(p, tokenizer) for p in puzzles]
        eval_datasets[diff] = Dataset.from_list(data)

        if verbose:
            print(f"Created {len(data)} {diff} eval examples")

    # Also create a mixed eval set
    if verbose:
        print(f"\nGenerating {num_per_difficulty} mixed eval puzzles...")

    mixed_puzzles = generate_puzzles(
        num_puzzles=num_per_difficulty,
        difficulty="mixed",
        seed=base_seed + 3000,
        verbose=verbose
    )
    mixed_data = [puzzle_to_dict(p, tokenizer) for p in mixed_puzzles]
    eval_datasets["mixed"] = Dataset.from_list(mixed_data)

    return eval_datasets


def create_full_dataset(
    train_examples: int = 10000,
    eval_per_difficulty: int = 200,
    train_seed: int = 42,
    eval_base_seed: int = 1000,
    tokenizer=None,
    verbose: bool = True
) -> DatasetDict:
    """
    Create full dataset with train and eval splits.

    Args:
        train_examples: Number of training examples
        eval_per_difficulty: Number of eval examples per difficulty
        train_seed: Random seed for training data
        eval_base_seed: Base seed for eval data
        tokenizer: Optional tokenizer
        verbose: Print progress

    Returns:
        DatasetDict with "train", "eval_easy", "eval_medium", "eval_hard", "eval_mixed"
    """
    # Create training dataset
    train_ds = create_training_dataset(
        num_examples=train_examples,
        difficulty="mixed",
        seed=train_seed,
        tokenizer=tokenizer,
        verbose=verbose
    )

    # Create eval datasets
    eval_datasets = create_eval_datasets(
        num_per_difficulty=eval_per_difficulty,
        base_seed=eval_base_seed,
        tokenizer=tokenizer,
        verbose=verbose
    )

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "train": train_ds,
        "eval_easy": eval_datasets["easy"],
        "eval_medium": eval_datasets["medium"],
        "eval_hard": eval_datasets["hard"],
        "eval_mixed": eval_datasets["mixed"],
    })

    return dataset_dict


def save_dataset(
    dataset: DatasetDict,
    output_dir: str = "data/hitoridata",
    save_to_hub: bool = False,
    hub_repo: Optional[str] = None,
    verbose: bool = True
):
    """
    Save dataset to disk and optionally to HuggingFace Hub.

    Args:
        dataset: DatasetDict to save
        output_dir: Local directory to save to
        save_to_hub: Whether to push to HuggingFace Hub
        hub_repo: Hub repository name (e.g., "username/hitori-puzzles")
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save to disk
    if verbose:
        print(f"\nSaving dataset to {output_dir}...")

    dataset.save_to_disk(output_dir)

    # Also save as JSON for easy inspection
    for split_name, split_data in dataset.items():
        json_path = os.path.join(output_dir, f"{split_name}.json")
        with open(json_path, "w") as f:
            # Convert to list of dicts
            data_list = [dict(row) for row in split_data]
            json.dump(data_list, f, indent=2)
        if verbose:
            print(f"  Saved {split_name}: {len(split_data)} examples")

    # Push to Hub
    if save_to_hub and hub_repo:
        if verbose:
            print(f"\nPushing to HuggingFace Hub: {hub_repo}")
        dataset.push_to_hub(hub_repo)


def load_dataset_from_disk(data_dir: str = "data/hitoridata") -> DatasetDict:
    """
    Load dataset from disk.

    Args:
        data_dir: Directory containing saved dataset

    Returns:
        DatasetDict
    """
    from datasets import load_from_disk
    return load_from_disk(data_dir)


# =============================================================================
# Main Script for Dataset Generation
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Hitori puzzle datasets")
    parser.add_argument("--train-examples", type=int, default=10000,
                        help="Number of training examples")
    parser.add_argument("--eval-per-difficulty", type=int, default=200,
                        help="Number of eval examples per difficulty")
    parser.add_argument("--output-dir", type=str, default="data/hitoridata",
                        help="Output directory")
    parser.add_argument("--train-seed", type=int, default=42,
                        help="Random seed for training data")
    parser.add_argument("--eval-seed", type=int, default=1000,
                        help="Base seed for eval data")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default=None,
                        help="HuggingFace Hub repo name")

    args = parser.parse_args()

    print("=" * 60)
    print("Hitori Dataset Generation")
    print("=" * 60)
    print(f"Training examples: {args.train_examples}")
    print(f"Eval examples per difficulty: {args.eval_per_difficulty}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create dataset
    dataset = create_full_dataset(
        train_examples=args.train_examples,
        eval_per_difficulty=args.eval_per_difficulty,
        train_seed=args.train_seed,
        eval_base_seed=args.eval_seed,
        verbose=True
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    # Save
    save_dataset(
        dataset,
        output_dir=args.output_dir,
        save_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        verbose=True
    )

    # Show sample
    print("\n" + "=" * 60)
    print("Sample training example:")
    print("-" * 60)
    sample = dataset["train"][0]
    print(f"Difficulty: {sample['difficulty']}")
    print(f"Target: {sample['target']}")
    print(f"\nPrompt (first 500 chars):")
    print(sample['prompt'][:500] + "...")
