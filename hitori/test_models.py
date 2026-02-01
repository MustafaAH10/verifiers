#!/usr/bin/env python3
"""
Test script for evaluating base models on Hitori puzzles.

This script tests the zero-shot performance of LLMs on Hitori puzzles
before any RL training, to establish a baseline.

=============================================================================
INSTALLATION
=============================================================================

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install torch transformers accelerate
pip install huggingface_hub

# For faster inference (optional but recommended)
pip install flash-attn --no-build-isolation

# Login to HuggingFace (required for Llama models)
huggingface-cli login

=============================================================================
USAGE
=============================================================================

# Run with default settings (10 puzzles per model)
python test_models.py

# Run with more puzzles
python test_models.py --num-puzzles 50

# Test only one model
python test_models.py --models llama

# Use CPU (slow, not recommended)
python test_models.py --device cpu

=============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generator import generate_puzzle, format_r1_prompt, HitoriPuzzle
from solver import parse_coordinates, verify_solution, format_solution


# =============================================================================
# Model Configurations
# =============================================================================

MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "description": "Llama 3.2 3B Instruct",
        "torch_dtype": torch.bfloat16,
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen 2.5 7B Instruct",
        "torch_dtype": torch.bfloat16,
    },
}


# =============================================================================
# Evaluation Functions
# =============================================================================

def extract_answer(completion: str) -> Optional[str]:
    """Extract the last <answer> tag content from completion."""
    # Add the think prefix since it's part of the prompt
    full_completion = "<think>" + completion

    matches = re.findall(r"<answer>(.*?)</answer>", full_completion, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


def evaluate_completion(
    completion: str,
    puzzle_grid: List[List[int]],
    expected_solution: Set[Tuple[int, int]]
) -> Dict:
    """Evaluate a model completion against the puzzle."""
    result = {
        "format_correct": False,
        "solution_correct": False,
        "has_think_close": False,
        "has_answer": False,
        "proposed_solution": None,
        "expected_solution": format_solution(expected_solution),
        "verification_details": {},
    }

    full_completion = "<think>" + completion

    # Check format
    result["has_think_close"] = "</think>" in full_completion
    result["has_answer"] = re.search(r"<answer>[\s\S]*?</answer>", full_completion) is not None
    result["format_correct"] = result["has_think_close"] and result["has_answer"]

    # Extract and verify solution
    answer_text = extract_answer(completion)
    if answer_text:
        proposed_shaded = parse_coordinates(answer_text)
        if proposed_shaded:
            result["proposed_solution"] = format_solution(proposed_shaded)
            is_valid, details = verify_solution(puzzle_grid, proposed_shaded)
            result["solution_correct"] = is_valid
            result["verification_details"] = details

    return result


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    device: str = "cuda"
) -> str:
    """Generate a completion from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    input_length = inputs["input_ids"].shape[1]
    completion = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

    return completion


# =============================================================================
# Main Test Function
# =============================================================================

def test_model(
    model_key: str,
    num_puzzles: int = 10,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """Test a model on Hitori puzzles."""

    model_config = MODELS[model_key]
    model_name = model_config["name"]

    print(f"\n{'='*60}")
    print(f"Testing: {model_config['description']}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Puzzles: {num_puzzles}")
    print(f"{'='*60}\n")

    # Load model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_config["torch_dtype"],
        device_map=device if device == "auto" else None,
        trust_remote_code=True,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()

    print(f"Model loaded. Parameters: {model.num_parameters():,}\n")

    # Generate puzzles and test
    results = []
    format_correct = 0
    solution_correct = 0

    for i in range(num_puzzles):
        print(f"Puzzle {i+1}/{num_puzzles}...", end=" ", flush=True)

        # Generate puzzle
        puzzle = generate_puzzle(difficulty="medium")
        if puzzle is None:
            print("Failed to generate puzzle, skipping")
            continue

        # Format prompt
        prompt = format_r1_prompt(puzzle, tokenizer)

        # Generate completion
        start_time = time.time()
        completion = generate_completion(model, tokenizer, prompt, device=device)
        elapsed = time.time() - start_time

        # Evaluate
        eval_result = evaluate_completion(completion, puzzle.grid, puzzle.solution)
        eval_result["puzzle_grid"] = puzzle.grid
        eval_result["completion"] = completion
        eval_result["time_seconds"] = elapsed

        results.append(eval_result)

        if eval_result["format_correct"]:
            format_correct += 1
        if eval_result["solution_correct"]:
            solution_correct += 1

        status = "✓" if eval_result["solution_correct"] else ("F" if eval_result["format_correct"] else "X")
        print(f"{status} ({elapsed:.1f}s)")

        if verbose and not eval_result["solution_correct"]:
            print(f"    Expected: {eval_result['expected_solution']}")
            print(f"    Proposed: {eval_result['proposed_solution'] or 'N/A'}")
            if eval_result["verification_details"]:
                print(f"    Details: {eval_result['verification_details']}")

    # Summary
    summary = {
        "model": model_name,
        "model_key": model_key,
        "num_puzzles": num_puzzles,
        "format_correct": format_correct,
        "format_accuracy": format_correct / num_puzzles if num_puzzles > 0 else 0,
        "solution_correct": solution_correct,
        "solution_accuracy": solution_correct / num_puzzles if num_puzzles > 0 else 0,
        "results": results,
    }

    print(f"\n{'-'*40}")
    print(f"Results for {model_config['description']}:")
    print(f"  Format Accuracy:   {format_correct}/{num_puzzles} ({summary['format_accuracy']*100:.1f}%)")
    print(f"  Solution Accuracy: {solution_correct}/{num_puzzles} ({summary['solution_accuracy']*100:.1f}%)")
    print(f"{'-'*40}\n")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Test LLMs on Hitori puzzles")
    parser.add_argument("--num-puzzles", type=int, default=10,
                        help="Number of puzzles to test per model")
    parser.add_argument("--models", type=str, default="all",
                        choices=["all", "llama", "qwen"],
                        help="Which models to test")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for puzzle generation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Show detailed output for failed puzzles")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Select models to test
    if args.models == "all":
        models_to_test = list(MODELS.keys())
    else:
        models_to_test = [args.models]

    # Run tests
    all_results = {}

    for model_key in models_to_test:
        try:
            summary = test_model(
                model_key=model_key,
                num_puzzles=args.num_puzzles,
                device=args.device,
                seed=args.seed,
                verbose=args.verbose
            )
            all_results[model_key] = summary
        except Exception as e:
            print(f"Error testing {model_key}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"error": str(e)}

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    for model_key, result in all_results.items():
        if "error" in result:
            print(f"{MODELS[model_key]['description']}: ERROR - {result['error']}")
        else:
            print(f"{MODELS[model_key]['description']}:")
            print(f"  Format:   {result['format_accuracy']*100:.1f}%")
            print(f"  Solution: {result['solution_accuracy']*100:.1f}%")

    # Save results
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_results_{timestamp}.json"

    # Convert results to JSON-serializable format
    json_results = {}
    for model_key, result in all_results.items():
        if "error" in result:
            json_results[model_key] = result
        else:
            json_results[model_key] = {
                "model": result["model"],
                "num_puzzles": result["num_puzzles"],
                "format_accuracy": result["format_accuracy"],
                "solution_accuracy": result["solution_accuracy"],
                "format_correct": result["format_correct"],
                "solution_correct": result["solution_correct"],
            }

    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
