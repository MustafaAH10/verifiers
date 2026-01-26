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
# Rollout Logger - Systematic logging of puzzles, completions, and rewards
# =============================================================================

class RolloutLogger:
    """Logs rollouts systematically with puzzles, completions, and rewards."""

    def __init__(self, log_dir: str = "rollout_logs", samples_per_file: int = 100):
        self.log_dir = log_dir
        self.samples_per_file = samples_per_file
        self.current_samples = []
        self.file_counter = 0
        self.total_logged = 0
        os.makedirs(log_dir, exist_ok=True)

    def log_rollout(
        self,
        puzzle_grid: List[List[int]],
        completion: str,
        format_reward: float,
        solution_reward: float,
        expected_solution: str,
        proposed_solution: str = None,
        verification_details: dict = None
    ):
        """Log a single rollout with all relevant information."""
        # Format the grid nicely
        grid_str = self._format_grid(puzzle_grid)

        sample = {
            "index": self.total_logged,
            "grid": grid_str,
            "completion": completion,
            "format_reward": format_reward,
            "solution_reward": solution_reward,
            "total_reward": format_reward + solution_reward,
            "expected_solution": expected_solution,
            "proposed_solution": proposed_solution or "N/A",
            "verification": verification_details or {}
        }

        self.current_samples.append(sample)
        self.total_logged += 1

        # Write to file when we have enough samples
        if len(self.current_samples) >= self.samples_per_file:
            self._flush_to_file()

    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid for display."""
        if not grid:
            return "N/A"
        columns = "ABCDEF"
        lines = []
        lines.append("     " + "  ".join(columns[:len(grid)]))
        lines.append("   +" + "-" * (len(grid) * 3 - 1) + "+")
        for i, row in enumerate(grid):
            row_str = f"R{i+1} | " + "  ".join(str(x) for x in row) + " |"
            lines.append(row_str)
        lines.append("   +" + "-" * (len(grid) * 3 - 1) + "+")
        return "\n".join(lines)

    def _flush_to_file(self):
        """Write current samples to a file."""
        if not self.current_samples:
            return

        filename = os.path.join(
            self.log_dir,
            f"rollouts_{self.file_counter:04d}.txt"
        )

        with open(filename, "w") as f:
            f.write(f"# Rollout Log File {self.file_counter}\n")
            f.write(f"# Samples: {len(self.current_samples)}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            for sample in self.current_samples:
                f.write(f"{'='*80}\n")
                f.write(f"ROLLOUT #{sample['index']}\n")
                f.write(f"{'='*80}\n\n")

                f.write("PUZZLE:\n")
                f.write(sample['grid'] + "\n\n")

                f.write(f"EXPECTED SOLUTION: {sample['expected_solution']}\n")
                f.write(f"PROPOSED SOLUTION: {sample['proposed_solution']}\n\n")

                f.write(f"REWARDS:\n")
                f.write(f"  Format:   {sample['format_reward']:.1f}\n")
                f.write(f"  Solution: {sample['solution_reward']:.1f}\n")
                f.write(f"  Total:    {sample['total_reward']:.1f}\n\n")

                if sample['verification']:
                    f.write(f"VERIFICATION DETAILS:\n")
                    for k, v in sample['verification'].items():
                        f.write(f"  {k}: {v}\n")
                    f.write("\n")

                f.write("MODEL COMPLETION:\n")
                f.write("-" * 40 + "\n")
                f.write(sample['completion'] + "\n")
                f.write("-" * 40 + "\n\n")

        logger.info(f"Wrote {len(self.current_samples)} rollouts to {filename}")
        self.current_samples = []
        self.file_counter += 1

    def flush(self):
        """Force flush remaining samples to file."""
        if self.current_samples:
            self._flush_to_file()


# Global rollout logger instance
rollout_logger = RolloutLogger(log_dir="rollout_logs", samples_per_file=100)


# =============================================================================
# Script Arguments
# =============================================================================

@dataclass
class ScriptArguments:
    """Arguments specific to Hitori training."""
    dataset_path: str = "data/hitoridata"  # Path to generated dataset
    tokenizer_name_or_path: str = None
    max_train_samples: int = -1  # -1 for all
    log_samples_prob: float = 0.1  # Probability of logging samples

    # Wandb logging
    wandb_project: str = "hitori-grpo"  # Wandb project name
    wandb_run_name: str = None  # Wandb run name (auto-generated if None)
    wandb_tags: str = None  # Comma-separated tags for wandb


# =============================================================================
# Reward Functions with Systematic Logging
# =============================================================================

# Cache for format rewards to coordinate logging between reward functions
_format_reward_cache = {}
_log_sample_rate = 0.50  # Log 50% of samples


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
    global _format_reward_cache
    rewards = []

    for completion in completions:
        try:
            # Add synthetic <think> as it's already part of the prompt
            full_completion = "<think>" + completion

            # Check format: must have </think> followed eventually by <answer>...</answer>
            has_think_close = "</think>" in full_completion
            has_answer = re.search(r"<answer>[\s\S]*?</answer>", full_completion) is not None

            reward = 1.0 if (has_think_close and has_answer) else 0.0
            rewards.append(reward)

            # Cache for later use by solution_reward_func
            _format_reward_cache[id(completion)] = reward

        except Exception as e:
            logger.debug(f"Format reward error: {e}")
            rewards.append(0.0)
            _format_reward_cache[id(completion)] = 0.0

    return rewards


def solution_reward_func(
    completions: List[str],
    target: List[str],
    grid: List[List[List[int]]],
    solution: List[List[List[int]]],
    **kwargs
) -> List[float]:
    """
    Reward function for solution correctness with systematic logging.

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
    global _format_reward_cache, rollout_logger
    rewards = []

    for completion, gt_target, puzzle_grid, expected_solution in zip(
        completions, target, grid, solution
    ):
        try:
            # Add synthetic <think> prefix
            full_completion = "<think>" + completion

            # Get format reward from cache
            format_reward = _format_reward_cache.pop(id(completion), 0.0)

            # Extract the LAST <answer> tag (final answer only)
            matches = re.findall(r"<answer>(.*?)<\/answer>", full_completion, re.DOTALL)

            proposed_solution_str = "No answer tag found"
            verification_details = {}
            solution_reward = 0.0

            if matches:
                # Use the last answer (the final one)
                answer_text = matches[-1].strip()

                # Parse coordinates from the answer
                proposed_shaded = parse_coordinates(answer_text)

                if proposed_shaded:
                    proposed_solution_str = format_solution(proposed_shaded)

                    # Verify the solution
                    is_valid, details = verify_solution(puzzle_grid, proposed_shaded)
                    verification_details = details

                    if is_valid:
                        solution_reward = 1.0
                else:
                    proposed_solution_str = f"Could not parse: {answer_text[:50]}..."
                    verification_details = {"error": "Failed to parse coordinates"}

            rewards.append(solution_reward)

            # Log rollout with probability
            if random.random() < _log_sample_rate:
                try:
                    rollout_logger.log_rollout(
                        puzzle_grid=puzzle_grid,
                        completion=full_completion,
                        format_reward=format_reward,
                        solution_reward=solution_reward,
                        expected_solution=gt_target,
                        proposed_solution=proposed_solution_str,
                        verification_details=verification_details
                    )
                except Exception as log_error:
                    logger.debug(f"Rollout logging error: {log_error}")

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

    # =========================================================================
    # Wandb Setup
    # =========================================================================
    if training_args.report_to and "wandb" in training_args.report_to:
        try:
            import wandb

            # Set wandb project
            os.environ["WANDB_PROJECT"] = script_args.wandb_project

            # Generate run name if not provided
            run_name = script_args.wandb_run_name
            if run_name is None:
                run_name = f"hitori-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # Parse tags
            tags = ["hitori", "grpo", "puzzle"]
            if script_args.wandb_tags:
                tags.extend([t.strip() for t in script_args.wandb_tags.split(",")])

            # Initialize wandb if not already initialized
            if wandb.run is None:
                wandb.init(
                    project=script_args.wandb_project,
                    name=run_name,
                    tags=tags,
                    config={
                        "model": model_args.model_name_or_path,
                        "dataset": script_args.dataset_path,
                        "max_train_samples": script_args.max_train_samples,
                    },
                )
            logger.info(f"Wandb initialized: project={script_args.wandb_project}, run={run_name}")

        except ImportError:
            logger.warning("wandb not installed. Skipping wandb logging.")
            training_args.report_to = [r for r in training_args.report_to if r != "wandb"]

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

    # Finish wandb run
    if training_args.report_to and "wandb" in training_args.report_to:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                logger.info("Wandb run finished")
        except ImportError:
            pass

    # Flush any remaining rollout logs
    rollout_logger.flush()
    logger.info(f"Total rollouts logged: {rollout_logger.total_logged}")

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
