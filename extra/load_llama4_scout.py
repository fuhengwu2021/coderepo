#!/usr/bin/env python3
"""
Direct model loading script for Llama-4-Scout
Tests loading the model with vLLM or SGLang directly (without Kubernetes)
"""

import argparse
import sys
import os

# Model path
MODEL_PATH = "/mnt/co-research/shared-models/hub/models--meta-llama--Llama-4-Scout-17B-16E-Instruct/snapshots/4bd10c4dc905b4000d76640d07a552344146faec"


def load_with_vllm(max_model_len: int = 2097152, tensor_parallel_size: int = 8):
    """Load model with vLLM"""
    try:
        from vllm import LLM, SamplingParams
        print("✅ vLLM imported successfully")
    except ImportError:
        print("❌ Error: vLLM not installed. Install with: pip install vllm")
        return False
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Configuration:")
    print(f"  - Max model length: {max_model_len} tokens (2M)")
    print(f"  - Tensor parallel size: {tensor_parallel_size} (8x H200)")
    print()
    
    try:
        print("🔄 Initializing vLLM engine...")
        llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
        print("✅ Model loaded successfully!")
        
        # Test with a simple prompt
        print("\n🧪 Testing with a simple prompt...")
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=200,
        )
        
        prompt = "Hello, how are you? Please provide a brief response."
        outputs = llm.generate([prompt], sampling_params)
        
        generated_text = outputs[0].outputs[0].text
        print(f"✅ Generation successful!")
        print(f"📝 Generated text: {generated_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_with_sglang(context_length: int = 2097152, tp: int = 8):
    """Load model with SGLang"""
    try:
        import sglang as sgl
        print("✅ SGLang imported successfully")
    except ImportError:
        print("❌ Error: SGLang not installed. Install with: pip install sglang[all]")
        return False
    
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Configuration:")
    print(f"  - Context length: {context_length} tokens (2M)")
    print(f"  - Tensor parallel size: {tp} (8x H200)")
    print()
    
    try:
        print("🔄 Initializing SGLang runtime...")
        runtime = sgl.Runtime(
            model_path=MODEL_PATH,
            tp=tp,
            context_length=context_length,
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully!")
        
        # Test with a simple prompt
        print("\n🧪 Testing with a simple prompt...")
        prompt = "Hello, how are you? Please provide a brief response."
        
        state = runtime.get_state()
        state.append("user", prompt)
        state.append("assistant", "")
        
        output = state.generate(
            max_new_tokens=200,
            temperature=0.7,
        )
        
        print(f"✅ Generation successful!")
        print(f"📝 Generated text: {output[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Load Llama-4-Scout model directly with vLLM or SGLang"
    )
    parser.add_argument(
        "--backend",
        choices=["vllm", "sglang"],
        required=True,
        help="Backend to use (vllm or sglang)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2097152,
        help="Max model length for vLLM (default: 2097152 = 2M)"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=2097152,
        help="Context length for SGLang (default: 2097152 = 2M)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Tensor parallel size (default: 8 for 8x H200)"
    )
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model path does not exist: {MODEL_PATH}")
        sys.exit(1)
    
    print("=" * 60)
    print(f"Loading Llama-4-Scout with {args.backend.upper()}")
    print("=" * 60)
    print()
    
    if args.backend == "vllm":
        success = load_with_vllm(
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size
        )
    else:
        success = load_with_sglang(
            context_length=args.context_length,
            tp=args.tensor_parallel_size
        )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
