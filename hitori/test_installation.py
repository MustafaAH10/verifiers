#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.

Usage:
    python test_installation.py
"""

import sys

def check_import(module_name, package_name=None, min_version=None):
    """Check if a module can be imported and optionally verify version."""
    package_name = package_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "unknown")

        # Handle nested module imports
        if "." in module_name:
            for part in module_name.split(".")[1:]:
                module = getattr(module, part)

        print(f"  [OK] {package_name}: {version}")
        return True, version
    except ImportError as e:
        print(f"  [FAIL] {package_name}: {e}")
        return False, None
    except Exception as e:
        print(f"  [WARN] {package_name}: imported but error getting version: {e}")
        return True, "unknown"


def check_torch():
    """Check PyTorch installation and CUDA availability."""
    print("\n" + "="*60)
    print("PyTorch & CUDA")
    print("="*60)

    ok, version = check_import("torch")
    if not ok:
        return False

    import torch

    print(f"  [INFO] PyTorch version: {torch.__version__}")
    print(f"  [INFO] CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  [INFO] CUDA version: {torch.version.cuda}")
        print(f"  [INFO] cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  [INFO] GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  [INFO] GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("  [WARN] CUDA not available - training will be slow!")

    return True


def check_flash_attention():
    """Check Flash Attention installation."""
    print("\n" + "="*60)
    print("Flash Attention")
    print("="*60)

    try:
        import flash_attn
        print(f"  [OK] flash_attn: {flash_attn.__version__}")
        return True
    except ImportError:
        print("  [WARN] flash_attn not installed - will use standard attention")
        print("  [INFO] Install with: pip install flash-attn --no-build-isolation")
        return False


def check_transformers():
    """Check Transformers and related libraries."""
    print("\n" + "="*60)
    print("HuggingFace Libraries")
    print("="*60)

    results = []
    results.append(check_import("transformers")[0])
    results.append(check_import("datasets")[0])
    results.append(check_import("accelerate")[0])
    results.append(check_import("peft")[0])
    results.append(check_import("huggingface_hub")[0])

    return all(results)


def check_trl():
    """Check TRL installation."""
    print("\n" + "="*60)
    print("TRL (Training Library)")
    print("="*60)

    ok, version = check_import("trl")
    if not ok:
        return False

    # Check GRPOTrainer is available
    try:
        from trl import GRPOTrainer, GRPOConfig
        print("  [OK] GRPOTrainer available")
    except ImportError as e:
        print(f"  [FAIL] GRPOTrainer not available: {e}")
        return False

    return True


def check_vllm():
    """Check vLLM installation."""
    print("\n" + "="*60)
    print("vLLM (Fast Generation)")
    print("="*60)

    ok, version = check_import("vllm")
    if not ok:
        print("  [WARN] vLLM not installed - generation will be slower")
        print("  [INFO] Install with: pip install vllm>=0.7.0")
        return False

    return True


def check_deepspeed():
    """Check DeepSpeed installation."""
    print("\n" + "="*60)
    print("DeepSpeed (Distributed Training)")
    print("="*60)

    ok, version = check_import("deepspeed")
    if not ok:
        print("  [WARN] DeepSpeed not installed - multi-GPU training may not work")
        return False

    # Check DeepSpeed ops
    try:
        import deepspeed
        print(f"  [INFO] DeepSpeed ops path: {deepspeed.ops.__path__}")
    except Exception as e:
        print(f"  [WARN] Could not check DeepSpeed ops: {e}")

    return True


def check_quantization():
    """Check quantization libraries."""
    print("\n" + "="*60)
    print("Quantization Support")
    print("="*60)

    ok, _ = check_import("bitsandbytes")
    if not ok:
        print("  [WARN] bitsandbytes not installed - 4/8-bit quantization unavailable")

    return ok


def check_logging():
    """Check logging libraries."""
    print("\n" + "="*60)
    print("Logging & Tracking")
    print("="*60)

    results = []
    results.append(check_import("wandb")[0])
    results.append(check_import("tensorboard")[0])

    return all(results)


def check_local_modules():
    """Check local Hitori modules."""
    print("\n" + "="*60)
    print("Local Hitori Modules")
    print("="*60)

    results = []

    try:
        from solver import parse_coordinates, verify_solution
        print("  [OK] solver module")
        results.append(True)
    except ImportError as e:
        print(f"  [FAIL] solver module: {e}")
        results.append(False)

    try:
        from generator import generate_puzzles, HitoriPuzzle
        print("  [OK] generator module")
        results.append(True)
    except ImportError as e:
        print(f"  [FAIL] generator module: {e}")
        results.append(False)

    return all(results)


def run_simple_test():
    """Run a simple end-to-end test."""
    print("\n" + "="*60)
    print("Simple Functional Test")
    print("="*60)

    try:
        # Test puzzle generation
        from generator import generate_puzzles
        puzzles = generate_puzzles(num_puzzles=1, difficulty="easy", seed=42, verbose=False)
        print(f"  [OK] Generated 1 puzzle")

        # Test solution verification
        from solver import verify_solution
        puzzle = puzzles[0]
        is_valid, details = verify_solution(puzzle.grid, puzzle.solution)
        print(f"  [OK] Solution verification: valid={is_valid}")

        # Test tokenizer loading (if model exists)
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
            print(f"  [OK] Tokenizer loads from HF Hub")
        except Exception as e:
            print(f"  [WARN] Could not load tokenizer from HF Hub: {e}")

        return True
    except Exception as e:
        print(f"  [FAIL] Functional test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("Hitori GRPO Training - Dependency Check")
    print("="*60)
    print(f"Python: {sys.version}")

    results = {}

    # Core dependencies
    results["torch"] = check_torch()
    results["flash_attn"] = check_flash_attention()
    results["transformers"] = check_transformers()
    results["trl"] = check_trl()
    results["vllm"] = check_vllm()
    results["deepspeed"] = check_deepspeed()
    results["quantization"] = check_quantization()
    results["logging"] = check_logging()
    results["local"] = check_local_modules()
    results["functional"] = run_simple_test()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    critical = ["torch", "transformers", "trl", "local"]
    recommended = ["vllm", "deepspeed", "flash_attn"]
    optional = ["quantization", "logging"]

    all_critical_ok = all(results.get(k, False) for k in critical)
    all_recommended_ok = all(results.get(k, False) for k in recommended)

    print("\nCritical (required):")
    for k in critical:
        status = "OK" if results.get(k, False) else "MISSING"
        print(f"  {k}: {status}")

    print("\nRecommended (for full performance):")
    for k in recommended:
        status = "OK" if results.get(k, False) else "MISSING"
        print(f"  {k}: {status}")

    print("\nOptional:")
    for k in optional:
        status = "OK" if results.get(k, False) else "MISSING"
        print(f"  {k}: {status}")

    print("\n" + "="*60)
    if all_critical_ok and all_recommended_ok:
        print("All dependencies installed correctly!")
        print("Ready for distributed training with vLLM.")
        return 0
    elif all_critical_ok:
        print("Critical dependencies OK, but some recommended packages missing.")
        print("Training will work but may be slower.")
        return 0
    else:
        print("CRITICAL dependencies missing! Please install them first.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
