"""
GRPO Training Script for Hitori Puzzles.

Based on the mini-deepseek-r1-aha-grpo training pattern.
Uses TRL's GRPOTrainer with vLLM for fast generation.

Usage:
    # Single GPU with Q-LoRA (slow, for testing)
    python train_hitori_grpo.py

    # Multi-GPU with DeepSpeed and vLLM
    accelerate launch --num_processes 3 --config_file configs/deepspeed_zero3.yaml \
        train_hitori_grpo.py --config configs/hitori_grpo.yaml
"""

import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Set, Tuple

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer, get_peft_config, ModelConfig, TrlParser

# Local imports
from solver import parse_coordinates, verify_solution, format_solution

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


# =============================================================================
# Script Arguments
# =============================================================================

@dataclass
class ScriptArguments:
    """Arguments specific to Hitori training."""
    dataset_path: str = "data/hitori"  # Path to generated dataset
    tokenizer_name_or_path: str = None
    max_train_samples: int = -1  # -1 for all
    log_samples_prob: float = 0.1  # Probability of logging samples


# =============================================================================
# Reward Functions
# =============================================================================

def format_reward_func(completions: List[str], target: List[str], **kwargs) -> List[float]:
    """
    Reward function for format compliance.

    Checks if the completion follows the expected format:
    <think>...</think>
    <answer>...</answer>

    Args:
        completions: List of model completions
        target: List of target solutions (not used here)

    Returns:
        List of reward scores (0.0 or 1.0)
    """
    rewards = []

    for completion in completions:
        try:
            # Add synthetic <think> as it's already part of the prompt
            completion = "<think>" + completion

            # Log some samples for debugging
            if random.random() < 0.1:
                os.makedirs("completion_samples", exist_ok=True)
                log_file = os.path.join("completion_samples", "hitori_completion_samples.txt")
                with open(log_file, "a") as f:
                    f.write(f"\n\n{'='*60}\n")
                    f.write(completion)

            # Check format: <think>...</think>\n<answer>...</answer>
            # Allow for flexible whitespace and content
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*?)<\/answer>"

            match = re.search(regex, completion, re.DOTALL)

            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)

        except Exception as e:
            logger.debug(f"Format reward error: {e}")
            rewards.append(0.0)

    return rewards


def solution_reward_func(
    completions: List[str],
    target: List[str],
    grid: List[List[List[int]]],
    solution: List[List[List[int]]],
    **kwargs
) -> List[float]:
    """
    Reward function for solution correctness.

    Checks if the proposed solution satisfies all Hitori constraints:
    1. Uniqueness: each number appears once per row/column (unshaded)
    2. Non-adjacency: no two shaded cells are adjacent
    3. Connectivity: all unshaded cells form a connected region

    Args:
        completions: List of model completions
        target: List of target solutions as coordinate strings
        grid: List of puzzle grids
        solution: List of expected solutions as coordinate lists

    Returns:
        List of reward scores (0.0 or 1.0)
    """
    rewards = []

    for completion, gt_target, puzzle_grid, expected_solution in zip(
        completions, target, grid, solution
    ):
        try:
            # Add synthetic <think> prefix
            completion = "<think>" + completion

            # Extract answer from tags
            match = re.search(r"<answer>(.*?)<\/answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(0.0)
                continue

            answer_text = match.group(1).strip()

            # Parse coordinates from the answer
            proposed_shaded = parse_coordinates(answer_text)

            if not proposed_shaded:
                rewards.append(0.0)
                continue

            # Verify the solution
            is_valid, details = verify_solution(puzzle_grid, proposed_shaded)

            if is_valid:
                rewards.append(1.0)

                # Log successful solutions
                if random.random() < 0.1:
                    os.makedirs("completion_samples", exist_ok=True)
                    log_file = os.path.join("completion_samples", "hitori_success_samples.txt")
                    with open(log_file, "a") as f:
                        f.write(f"\n\n{'='*60}\n")
                        f.write(f"Grid: {puzzle_grid}\n")
                        f.write(f"Proposed: {format_solution(proposed_shaded)}\n")
                        f.write(f"Expected: {gt_target}\n")
                        f.write(f"Completion:\n{completion}\n")
            else:
                rewards.append(0.0)

        except Exception as e:
            logger.debug(f"Solution reward error: {e}")
            rewards.append(0.0)

    return rewards


# =============================================================================
# Dataset Preparation
# =============================================================================

def prepare_dataset(tokenizer, script_args: ScriptArguments) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare the Hitori dataset for GRPO training.

    Args:
        tokenizer: Tokenizer for prompt formatting
        script_args: Script arguments

    Returns:
        (train_dataset, test_dataset)
    """
    # Try loading from local path first
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(script_args.dataset_path)
        train_dataset = dataset["train"]
        test_dataset = dataset.get("eval_mixed", dataset.get("eval_medium", train_dataset))
    except Exception:
        # Generate on the fly if no pre-generated dataset
        logger.info("No pre-generated dataset found. Generating puzzles on the fly...")
        from generator import generate_puzzles, format_r1_prompt

        # Generate training puzzles
        train_puzzles = generate_puzzles(
            num_puzzles=min(5000, script_args.max_train_samples if script_args.max_train_samples > 0 else 5000),
            difficulty="mixed",
            seed=42,
            verbose=True
        )

        # Generate test puzzles
        test_puzzles = generate_puzzles(
            num_puzzles=500,
            difficulty="mixed",
            seed=1234,
            verbose=True
        )

        def puzzle_to_dict(puzzle):
            return {
                "prompt": format_r1_prompt(puzzle, tokenizer),
                "target": format_solution(puzzle.solution),
                "grid": puzzle.grid,
                "solution": [list(coord) for coord in puzzle.solution],
                "difficulty": puzzle.difficulty,
            }

        train_dataset = Dataset.from_list([puzzle_to_dict(p) for p in train_puzzles])
        test_dataset = Dataset.from_list([puzzle_to_dict(p) for p in test_puzzles])

    # Subset if requested
    if script_args.max_train_samples > 0:
        train_dataset = train_dataset.shuffle(seed=42).select(
            range(min(script_args.max_train_samples, len(train_dataset)))
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    return train_dataset, test_dataset


def get_checkpoint(training_args: GRPOConfig):
    """Get last checkpoint if resuming training."""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


# =============================================================================
# Main Training Function
# =============================================================================

def grpo_function(
    model_args: ModelConfig,
    script_args: ScriptArguments,
    training_args: GRPOConfig
):
    """
    Main GRPO training function for Hitori.

    Args:
        model_args: Model configuration
        script_args: Script-specific arguments
        training_args: GRPO training configuration
    """
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"Script parameters: {script_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name_or_path
        if script_args.tokenizer_name_or_path
        else model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare datasets
    train_dataset, test_dataset = prepare_dataset(tokenizer, script_args)

    # Create trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[format_reward_func, solution_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )

    # Check for checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")

    # Train
    logger.info(
        f"*** Starting training {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"for {training_args.num_train_epochs} epochs ***"
    )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    # Save model
    logger.info("*** Saving model ***")
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")

    # Create model card
    if trainer.accelerator.is_main_process:
        trainer.create_model_card({"tags": ["rl", "grpo", "hitori", "puzzle"]})

    # Push to hub if requested
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

    logger.info("*** Training complete! ***")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for training."""
    parser = TrlParser((ModelConfig, ScriptArguments, GRPOConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run training
    grpo_function(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
