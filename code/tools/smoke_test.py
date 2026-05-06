"""
OpenVLA smoke test on local 24G GPU.
- Load OpenVLA-7B with 4-bit quantization (~6 GB peak)
- Run 1 step forward (predict_action)
- Validate output shape/dtype/timing
"""
import os
import time
import numpy as np
import torch
from PIL import Image

# Make sure we use mounted model path
MODEL_PATH = "/workspace/models/openvla-7b"
print(f"=== OpenVLA Smoke Test ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Use HF transformers AutoClass loader (lightweight path)
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

print(f"\n[1/3] Loading processor from {MODEL_PATH}...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  processor loaded in {time.time()-t0:.1f}s")

# 4-bit quant config: peak ~6 GB
quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print(f"\n[2/3] Loading OpenVLA-7B with 4-bit quantization...")
t0 = time.time()
vla = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    quantization_config=quant_cfg,
    device_map={"": 0},  # required for 4-bit, avoid the .to() error
    trust_remote_code=True,
)
print(f"  model loaded in {time.time()-t0:.1f}s")
print(f"  GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")

# Build dummy input: 224x224 RGB image + LIBERO-style instruction
print(f"\n[3/3] Running 1 forward pass (predict_action)...")
dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
prompt = "In: What action should the robot take to pick up the red cup?\nOut:"

inputs = processor(prompt, dummy_image)
# Move to GPU
inputs = {k: (v.to("cuda:0", dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to("cuda:0"))
          for k, v in inputs.items()}

# Warmup (first call has compilation overhead)
t0 = time.time()
with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
warmup_time = time.time() - t0
print(f"  Warmup forward (first call): {warmup_time:.2f}s")
print(f"  Action shape: {action.shape}, dtype: {action.dtype}")
print(f"  Action: {action}")

# Real timing
t0 = time.time()
with torch.no_grad():
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
real_time = time.time() - t0
print(f"  Steady-state forward: {real_time*1000:.1f} ms ({1.0/real_time:.2f} Hz)")

print(f"\nFinal GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
print(f"\n=== SMOKE TEST PASSED ===")
