# vla-lab-v1.0 Container Usage Guide

Single docker image for the v1.5 paper: OpenVLA + Spirit v1.5 + π0.5
inference, training, and LIBERO evaluation. Built on top of
`spirit-v1.1-cu128-py311` with the additional dependencies needed for
all three VLA bases.

## Pull

```bash
docker pull micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311
```

Alternative registries (same image, same digest):

```bash
docker pull test-lab-instance-cn-beijing.cr.volces.com/evad-infra-compute/planningmodel:vla-lab-v1.0-cu128-py311
docker pull evad-ml-cn-bjdb-1-vecps.cr.cloud.vnet.com/infra_public/planning:vla-lab-v1.0-cu128-py311
```

Image facts:

- Size: 22.4 GB (only +0.2 GB on top of spirit-v1.1)
- Digest: sha256:4ac541a377810e6bd5af7620b21e8b5428103493afe7e0192a8c5929179f1d7d
- Base: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
- Python: 3.11

Key deps: torch 2.8 / transformers 4.57.1 / **timm 0.9.10** / **peft
0.19.1** / accelerate 1.0 / mani_skill 3.0.1 / sapien 3.0.3 / cv2
4.13 / libero (vendored). NO TensorFlow — see "Why no TF" below.

## Quick smoke tests

### Test 1: LIBERO env works

No model needed. Verifies osmesa CPU rendering + libero submodule.

```bash
docker run --rm --gpus all \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311 \
    python -c "
import os; os.environ['MUJOCO_GL']='osmesa'
import torch as _t
_t.load = (lambda f, *a, **kw: __import__('torch').serialization.load(f, *a, **{**kw, 'weights_only': False}))
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()['libero_spatial']()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)
env.reset()
print('LIBERO OK:', task.language)
"
```

### Test 2: OpenVLA loads

Requires the `openvla/openvla-7b-finetuned-libero-spatial` checkpoint
(~15 GB) downloaded locally and your `vla-lab` repo cloned.

```bash
docker run --rm --gpus all \
    -v /path/to/openvla:/openvla:ro \
    -v /path/to/openvla-7b-finetuned-libero-spatial:/model:ro \
    -v /path/to/vla-lab/code:/code:ro \
    -e PYTHONPATH=/code:/openvla \
    micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311 \
    python -c "
import sys; sys.path.insert(0, '/code'); sys.path.insert(0, '/openvla')
from post_training.adapters.openvla import OpenVLAAdapter
OpenVLAAdapter._patch_openvla_compat()
from transformers import AutoModelForVision2Seq
import torch, time
t0 = time.time()
model = AutoModelForVision2Seq.from_pretrained(
    '/model', torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
).to('cuda')
print(f'OpenVLA loaded in {time.time()-t0:.1f}s, mem={torch.cuda.memory_allocated()/1e9:.1f}GB')
print(f'unnorm_keys: {list(model.norm_stats.keys())}')
"
```

Expected output:

```
OpenVLA loaded in 17.3s, mem=15.1GB
unnorm_keys: ['libero_spatial']
```

### Test 3: One-shot LIBERO Spatial rollout

Full integration test: OpenVLA + LIBERO env + our adapter pipeline.

```bash
docker run --rm --gpus all \
    -v /path/to/openvla:/openvla:ro \
    -v /path/to/openvla-7b-finetuned-libero-spatial:/model:ro \
    -v /path/to/vla-lab/code:/code:ro \
    -v /tmp/eval_smoke:/output \
    -e PYTHONPATH=/code:/openvla \
    -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
    micr.cloud.mioffice.cn/world-model-lyk/planningmodel:vla-lab-v1.0-cu128-py311 \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /model \
        --suites libero_spatial \
        --n_tasks 1 --n_trials_per_task 5 --seeds 42 \
        --output_dir /output
```

Expected: 4/5 = 80% success on libero_spatial task 0, ~10 min wall clock
on RTX 3090 (BF16 inference, manual greedy decode).

## Running v1.5 paper experiments

### DPO training (offline pair-based)

```bash
# 1. Collect rollout pairs from a base ckpt
docker run --rm --gpus all ... \
    python -m post_training.rollout \
        --base openvla --base_ckpt /model \
        --suite libero_spatial \
        --n_tasks 10 --n_inits_per_task 5 --n_candidates_per_init 4 \
        --output_path /workspace/datasets/openvla_spatial_pairs.pt

# 2. Train DPO over those pairs
docker run --rm --gpus all ... \
    python -m post_training.train_dpo \
        --base openvla --base_ckpt /model --suite spatial \
        --pairs_file /workspace/datasets/openvla_spatial_pairs.pt \
        --output_dir /workspace/output/openvla_dpo_spatial \
        --max_steps 5000 --lora_r 32

# 3. Eval the trained ckpt on full 4-suite
docker run --rm --gpus all ... \
    python -m post_training.eval_libero \
        --base openvla --base_ckpt /model \
        --lora_ckpt /workspace/output/openvla_dpo_spatial/checkpoint-5000.pt \
        --suites libero_spatial libero_object libero_goal libero_10 \
        --n_trials_per_task 50 --seeds 42 1337 2026 \
        --output_dir /workspace/output/openvla_dpo_eval
```

### GRPO training (online K-rollout per step)

```bash
docker run --rm --gpus all ... \
    python -m post_training.train_grpo \
        --base openvla --base_ckpt /model --suite spatial \
        --max_steps 500 --group_size 4 \
        --output_dir /workspace/output/openvla_grpo_spatial
```

## Gotchas

### MUJOCO_GL=osmesa is mandatory

The container does NOT have NVIDIA Vulkan ICD mounted (toolkit's
default `compute,utility` capability omits graphics). MuJoCo must use
the osmesa CPU software renderer. Always set:

```
-e MUJOCO_GL=osmesa
-e PYOPENGL_PLATFORM=osmesa
```

Forgetting this gives a silent crash at `OffScreenRenderEnv()` construction.

### torch>=2.6 weights_only patch for LIBERO

LIBERO's init_states are torch pickles that contain numpy globals.
The default `weights_only=True` in torch>=2.6 refuses to load them.
Our rollout / eval code patches `torch.load` automatically. If running
arbitrary scripts that touch LIBERO directly, prepend:

```python
import torch as _t
_orig = _t.load
_t.load = lambda f, *a, **kw: _orig(f, *a, **{**kw, "weights_only": False})
```

### OpenVLA + transformers 4.57: MUST patch

The container ships transformers 4.57.1, which is incompatible with
OpenVLA's modeling_prismatic in two ways:

- Adds an `_supports_sdpa` check during `__init__` that the OpenVLA
  classes don't satisfy (it's a property delegating to `self.language_model`,
  which isn't yet assigned).
- `model.generate()` degenerates: returns bin 127 (action midpoint) for
  every action dim regardless of input → 0% LIBERO success.

Both fixes are baked into our `OpenVLAAdapter`:

```python
from post_training.adapters.openvla import OpenVLAAdapter
OpenVLAAdapter._patch_openvla_compat()  # call BEFORE loading the model
```

The adapter's `select_action` uses our `_manual_greedy_predict_action`
helper instead of `model.predict_action` (which calls the broken
`model.generate()`).

If you load OpenVLA outside our adapter, you need to call the patch
manually + reimplement greedy decode (~50 lines, see adapter source).

### Why no TF?

The v1.5 first-rollout debugging session found that importing
TensorFlow in the same Python process as LIBERO's `OffScreenRenderEnv`
causes a silent crash at env construction (likely TF's CUDA registration
interfering with mujoco's osmesa context init).

OpenVLA's official `get_vla_action` uses TF for image preprocessing
(`tf.image.encode_jpeg` + `tf.image.crop_and_resize`). Our adapter
substitutes:

- `tf.image.encode_jpeg` → PIL JPEG round-trip (~13 / 255 mean pixel diff)
- `tf.image.crop_and_resize` → cv2.warpAffine sub-pixel affine (~2.6 / 255 diff)

These approximations are close enough that we reproduce 80% spatial,
matching paper 84.7% / our v1.4 reproduction 78%.

## What's inside

```
/workspace/spirit-v1.5/      # Spirit v1.5 source (vendored from spirit-v1.1)
/workspace/LIBERO/           # LIBERO submodule (vendored)
/opt/vla-lab/                # bootstrap-ssh.sh helper from spirit-v1.1
/opt/conda/                  # python 3.11 + dependencies

User mounts (typical):
  /openvla       — OpenVLA repo (PYTHONPATH)
  /code          — vla-lab/code (PYTHONPATH)
  /model         — base model checkpoint
  /workspace/jfs — persistent storage for outputs
```

## Reproduce paper-grade evaluation

Full v1.5 main results table (paper §4.2) requires:

- 3 bases × 3 alg variants (sft/dpo/grpo) × 4 suites × 50 trials × 3 seeds
  = ~5400 episodes per base, ~16 200 total
- Per-episode: ~60–120 s on RTX 3090, ~30–60 s on H20

So full reproduction is ~270 GPU-hours on consumer cards or ~135 H-hours
on datacenter cards. For a quicker sanity check, drop trial count to 5
and seeds to 1 — that's ~1.5 h per (base, alg, suite) cell.

---

*Updated 2026-05-10 after fixing the transformers 4.57 generate
degeneration bug. See docs/v1.5_first_real_rollout.md for the full
incident analysis.*
