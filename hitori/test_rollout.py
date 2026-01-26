"""
Single Rollout Test Script for Hitori Puzzles.

Generates a puzzle and shows what the model output looks like for debugging
and understanding the training format.

Usage:
    # Test with base model
    python test_rollout.py --model-path Qwen/Qwen2.5-3B-Instruct

    # Test with trained model
    python test_rollout.py --model-path outputs/hitori-grpo

    # Test with specific difficulty
    python test_rollout.py --model-path models/qwen2.5-3b-instruct --difficulty hard
"""

import argparse
import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generator import generate_puzzle, format_r1_prompt
from solver import verify_solution, parse_coordinates, format_solution


def load_model(
    model_path: str,
    device: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """Load model and tokenizer."""
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if not (load_in_4bit or load_in_8bit) else None,
    }

    if device == "auto":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()

    return model, tokenizer


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """Generate a completion for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get only the generated part (exclude prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return completion


def evaluate_completion(completion: str, grid, expected_solution):
    """Evaluate the completion against the expected solution."""
    result = {
        "format_correct": False,
        "solution_correct": False,
        "proposed_solution": None,
        "error": None,
        "details": None,
    }

    try:
        # Add synthetic <think> prefix
        full_completion = "<think>" + completion

        # Check format
        format_regex = r"<think>.*</think>\s*<answer>(.*?)</answer>"
        match = re.search(format_regex, full_completion, re.DOTALL)

        if match:
            result["format_correct"] = True
            answer_text = match.group(1).strip()

            # Parse proposed solution
            proposed_shaded = parse_coordinates(answer_text)
            result["proposed_solution"] = format_solution(proposed_shaded) if proposed_shaded else "Failed to parse"

            if proposed_shaded:
                # Verify solution
                is_valid, details = verify_solution(grid, proposed_shaded)
                result["solution_correct"] = is_valid
                result["details"] = {
                    "uniqueness": details["uniqueness"][0],
                    "adjacency": details["adjacency"][0],
                    "connectivity": details["connectivity"][0],
                }

    except Exception as e:
        result["error"] = str(e)

    return result


def run_single_rollout(
    model_path: str,
    difficulty: str = "medium",
    seed: Optional[int] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    show_prompt: bool = True,
):
    """Run a single rollout and display results."""
    print("=" * 70)
    print("HITORI SINGLE ROLLOUT TEST")
    print("=" * 70)

    # Generate puzzle
    print(f"\nGenerating {difficulty} puzzle...")
    if seed is not None:
        import random
        random.seed(seed)

    puzzle = generate_puzzle(difficulty=difficulty)
    if puzzle is None:
        print("ERROR: Failed to generate puzzle")
        return

    print(f"\nPuzzle Grid:")
    print(puzzle.format_grid())
    print(f"\nExpected Solution: {puzzle.format_solution()}")
    print(f"Shaded cells: {len(puzzle.solution)}")

    # Load model
    print("\n" + "-" * 70)
    model, tokenizer = load_model(
        model_path,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Format prompt
    prompt = format_r1_prompt(puzzle, tokenizer)

    if show_prompt:
        print("\n" + "-" * 70)
        print("PROMPT:")
        print("-" * 70)
        print(prompt)

    # Generate completion
    print("\n" + "-" * 70)
    print(f"Generating completion (temp={temperature}, max_tokens={max_new_tokens})...")
    print("-" * 70)

    completion = generate_completion(
        model, tokenizer, prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    print("\nMODEL COMPLETION:")
    print("-" * 70)
    print(completion)

    # Evaluate
    print("\n" + "-" * 70)
    print("EVALUATION:")
    print("-" * 70)

    result = evaluate_completion(completion, puzzle.grid, puzzle.solution)

    print(f"Format Correct:   {'YES' if result['format_correct'] else 'NO'}")
    print(f"Solution Correct: {'YES' if result['solution_correct'] else 'NO'}")

    if result['proposed_solution']:
        print(f"\nProposed Solution: {result['proposed_solution']}")
        print(f"Expected Solution: {puzzle.format_solution()}")

    if result['details']:
        print(f"\nConstraint Check:")
        print(f"  Uniqueness:    {'PASS' if result['details']['uniqueness'] else 'FAIL'}")
        print(f"  Adjacency:     {'PASS' if result['details']['adjacency'] else 'FAIL'}")
        print(f"  Connectivity:  {'PASS' if result['details']['connectivity'] else 'FAIL'}")

    if result['error']:
        print(f"\nError: {result['error']}")

    # Show reward calculation
    print("\n" + "-" * 70)
    print("REWARD CALCULATION:")
    print("-" * 70)
    format_reward = 1.0 if result['format_correct'] else 0.0
    solution_reward = 1.0 if result['solution_correct'] else 0.0
    print(f"Format Reward:   {format_reward}")
    print(f"Solution Reward: {solution_reward}")
    print(f"Total Reward:    {format_reward + solution_reward}")

    print("\n" + "=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(description="Test single rollout for Hitori puzzle")
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model (local or HuggingFace)"
    )
    parser.add_argument(
        "--difficulty", type=str, default="medium",
        choices=["easy", "medium", "hard"],
        help="Puzzle difficulty"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for puzzle generation"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1024,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization"
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load model in 8-bit quantization"
    )
    parser.add_argument(
        "--hide-prompt", action="store_true",
        help="Don't show the full prompt"
    )

    args = parser.parse_args()

    run_single_rollout(
        model_path=args.model_path,
        difficulty=args.difficulty,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        show_prompt=not args.hide_prompt,
    )


if __name__ == "__main__":
    main()
