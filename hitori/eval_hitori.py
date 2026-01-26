"""
Hitori Puzzle Evaluation Script.

Evaluates trained models on Hitori puzzles with different difficulty levels.
Reports accuracy metrics and saves sample completions for analysis.

Usage:
    python eval_hitori.py \
        --model-path outputs/hitori-grpo \
        --eval-data data/hitori \
        --output-dir eval_results
"""

import argparse
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from solver import parse_coordinates, verify_solution, format_solution
from generator import format_r1_prompt, generate_puzzles


def load_model_and_tokenizer(
    model_path: str,
    device: str = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    """
    Load model and tokenizer from path.

    Args:
        model_path: Path to trained model or HuggingFace model name
        device: Device to load model on
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization

    Returns:
        (model, tokenizer)
    """
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
) -> str:
    """
    Generate a completion for a given prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Generated completion text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Get only the generated part (exclude prompt)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return completion


def evaluate_completion(
    completion: str,
    grid: List[List[int]],
    expected_solution: List[List[int]],
) -> Dict:
    """
    Evaluate a single completion against the expected solution.

    Args:
        completion: Model's completion text
        grid: The puzzle grid
        expected_solution: Expected solution as list of coordinate lists

    Returns:
        Dictionary with evaluation results
    """
    result = {
        "format_correct": False,
        "solution_correct": False,
        "uniqueness_satisfied": False,
        "adjacency_satisfied": False,
        "connectivity_satisfied": False,
        "proposed_solution": None,
        "error": None,
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
            result["proposed_solution"] = format_solution(proposed_shaded)

            if proposed_shaded:
                # Verify solution
                is_valid, details = verify_solution(grid, proposed_shaded)

                result["uniqueness_satisfied"] = details["uniqueness"][0]
                result["adjacency_satisfied"] = details["adjacency"][0]
                result["connectivity_satisfied"] = details["connectivity"][0]
                result["solution_correct"] = is_valid

    except Exception as e:
        result["error"] = str(e)

    return result


def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    max_samples: int = -1,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    verbose: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Evaluate model on a dataset.

    Args:
        model: The language model
        tokenizer: The tokenizer
        dataset: Dataset to evaluate
        max_samples: Maximum samples to evaluate (-1 for all)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        verbose: Show progress

    Returns:
        (list of results, aggregated metrics)
    """
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    results = []
    iterator = tqdm(dataset, desc="Evaluating") if verbose else dataset

    for example in iterator:
        prompt = example["prompt"]
        grid = example["grid"]
        expected_solution = example["solution"]
        difficulty = example.get("difficulty", "unknown")

        # Generate completion
        completion = generate_completion(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Evaluate
        eval_result = evaluate_completion(completion, grid, expected_solution)
        eval_result["difficulty"] = difficulty
        eval_result["expected_solution"] = example["target"]
        eval_result["completion"] = completion

        results.append(eval_result)

    # Aggregate metrics
    metrics = compute_metrics(results)

    return results, metrics


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute aggregated metrics from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary of metrics
    """
    total = len(results)
    if total == 0:
        return {}

    # Overall metrics
    metrics = {
        "total": total,
        "format_accuracy": sum(r["format_correct"] for r in results) / total,
        "solution_accuracy": sum(r["solution_correct"] for r in results) / total,
        "uniqueness_rate": sum(r["uniqueness_satisfied"] for r in results) / total,
        "adjacency_rate": sum(r["adjacency_satisfied"] for r in results) / total,
        "connectivity_rate": sum(r["connectivity_satisfied"] for r in results) / total,
    }

    # Per-difficulty metrics
    difficulty_results = defaultdict(list)
    for r in results:
        difficulty_results[r["difficulty"]].append(r)

    metrics["by_difficulty"] = {}
    for diff, diff_results in difficulty_results.items():
        diff_total = len(diff_results)
        metrics["by_difficulty"][diff] = {
            "total": diff_total,
            "format_accuracy": sum(r["format_correct"] for r in diff_results) / diff_total,
            "solution_accuracy": sum(r["solution_correct"] for r in diff_results) / diff_total,
        }

    return metrics


def save_results(
    results: List[Dict],
    metrics: Dict,
    output_dir: str,
    split_name: str = "eval",
):
    """
    Save evaluation results to files.

    Args:
        results: List of evaluation results
        metrics: Aggregated metrics
        output_dir: Output directory
        split_name: Name for this evaluation split
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(output_dir, f"{split_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save detailed results
    results_path = os.path.join(output_dir, f"{split_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save successful completions
    success_path = os.path.join(output_dir, f"{split_name}_successes.txt")
    with open(success_path, "w") as f:
        for r in results:
            if r["solution_correct"]:
                f.write("=" * 60 + "\n")
                f.write(f"Difficulty: {r['difficulty']}\n")
                f.write(f"Expected: {r['expected_solution']}\n")
                f.write(f"Proposed: {r['proposed_solution']}\n")
                f.write(f"Completion:\n{r['completion']}\n\n")

    # Save failures
    failure_path = os.path.join(output_dir, f"{split_name}_failures.txt")
    with open(failure_path, "w") as f:
        for r in results:
            if not r["solution_correct"]:
                f.write("=" * 60 + "\n")
                f.write(f"Difficulty: {r['difficulty']}\n")
                f.write(f"Format OK: {r['format_correct']}\n")
                f.write(f"Uniqueness: {r['uniqueness_satisfied']}\n")
                f.write(f"Adjacency: {r['adjacency_satisfied']}\n")
                f.write(f"Connectivity: {r['connectivity_satisfied']}\n")
                f.write(f"Expected: {r['expected_solution']}\n")
                f.write(f"Proposed: {r['proposed_solution']}\n")
                f.write(f"Completion:\n{r['completion']}\n\n")


def print_metrics(metrics: Dict):
    """Print metrics in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nTotal samples: {metrics['total']}")
    print(f"\nOverall Metrics:")
    print(f"  Format Accuracy:   {metrics['format_accuracy']*100:.1f}%")
    print(f"  Solution Accuracy: {metrics['solution_accuracy']*100:.1f}%")
    print(f"\nConstraint Satisfaction:")
    print(f"  Uniqueness:    {metrics['uniqueness_rate']*100:.1f}%")
    print(f"  Adjacency:     {metrics['adjacency_rate']*100:.1f}%")
    print(f"  Connectivity:  {metrics['connectivity_rate']*100:.1f}%")

    if "by_difficulty" in metrics:
        print(f"\nBy Difficulty:")
        for diff, diff_metrics in sorted(metrics["by_difficulty"].items()):
            print(f"  {diff.upper()}: {diff_metrics['solution_accuracy']*100:.1f}% "
                  f"({diff_metrics['total']} samples)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hitori puzzle solver")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--eval-data", type=str, default="data/hitori",
                        help="Path to evaluation dataset")
    parser.add_argument("--output-dir", type=str, default="eval_results",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Maximum samples per split (-1 for all)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit")
    parser.add_argument("--splits", type=str, nargs="+",
                        default=["eval_easy", "eval_medium", "eval_hard", "eval_mixed"],
                        help="Dataset splits to evaluate")

    args = parser.parse_args()

    print("=" * 60)
    print("Hitori Puzzle Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.eval_data}")
    print(f"Output: {args.output_dir}")
    print(f"Temperature: {args.temperature}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Load dataset
    print("Loading evaluation data...")
    dataset = load_from_disk(args.eval_data)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each split
    all_metrics = {}
    for split in args.splits:
        if split not in dataset:
            print(f"Warning: Split '{split}' not found in dataset")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating split: {split}")
        print(f"{'='*60}")

        split_data = dataset[split]
        results, metrics = evaluate_dataset(
            model, tokenizer, split_data,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Save results
        save_results(results, metrics, output_dir, split)
        print_metrics(metrics)

        all_metrics[split] = metrics

    # Save combined metrics
    combined_path = os.path.join(output_dir, "all_metrics.json")
    with open(combined_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
