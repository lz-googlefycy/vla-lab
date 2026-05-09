# Upstream contributions to Spirit v1.5

Four issues / PRs I've drafted for [Spirit-AI-Team/spirit-v1.5](https://github.com/Spirit-AI-Team/spirit-v1.5) after reproducing the model on consumer GPUs (RTX 3090) and a datacenter GPU server. Two are fixes, two are docs. They're independent — each can land alone.

This file has the full text so the community can comment / improve before submission, and so anyone hitting the same issues can use the bodies as starting points.

---

## #1 — feat: support `torch_dtype` and `device` in `from_pretrained`

**Title**: `feat: support torch_dtype + device params in SpiritVLAPolicy.from_pretrained`

**Body**:

> ### Background
>
> When deploying Spirit v1.5 on consumer GPUs (e.g. RTX 3090 24GB), the
> default load path puts the full 21.6 GB fp32 weights directly on
> CUDA via the `device` field in `config.json`. This OOMs immediately
> because PyTorch needs ~22 GB for the weights plus headroom.
>
> The standard HuggingFace `transformers` API resolves this by exposing
> `torch_dtype` and `device_map` as **call-time parameters**:
>
> ```python
> model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
> ```
>
> Spirit currently encodes "where to load" into `config.json["device"]`,
> which means consumer-GPU users must temporarily edit the ckpt file to
> deploy. This is a poor user experience and reverses the standard HF
> idiom.
>
> ### Proposed change
>
> Add optional `torch_dtype` and `device` parameters to
> `SpiritVLAPolicy.from_pretrained`:
>
> ```python
> model = SpiritVLAPolicy.from_pretrained(
>     ckpt_path,
>     torch_dtype=torch.bfloat16,
>     device="cuda",      # overrides config.json
>     train=False,
> )
> ```
>
> When the parameter is provided, it overrides `config.json["device"]`
> at load time and casts the state dict before placing it on device.
>
> ### Reference workaround in current downstream code
>
> Until this lands upstream, downstream code does this dance:
>
> ```python
> # hack to load 21GB fp32 weights on a 24GB consumer GPU
> with open(config_path, "r") as f: orig = f.read()
> cfg = json.loads(orig); cfg["device"] = "cpu"
> with open(config_path, "w") as f: json.dump(cfg, f, indent=2)
> try:
>     model = SpiritVLAPolicy.from_pretrained(ckpt_path, train=False)
> finally:
>     with open(config_path, "w") as f: f.write(orig)
> model = model.to(torch.bfloat16).to("cuda").eval()
> ```
>
> A first-class API parameter would remove the need for this.

---

## #2 — fix: parameterise dtype in `utils/sampling.py` for bf16/fp16 inference

**Title**: `fix(sampling): make sample_noise / sample_time follow autocast dtype`

**Body**:

> ### Bug
>
> `utils/sampling.py:sample_noise` and `sample_time` hardcode
> `dtype=torch.float32`:
>
> ```python
> def sample_noise(shape, device):
>     return torch.normal(
>         mean=torch.zeros(shape, dtype=torch.float32, device=device),
>         std=torch.ones(shape, dtype=torch.float32, device=device),
>     )
> ```
>
> This is fine when DiT runs in fp32, but breaks when running in
> bf16 (e.g. for consumer-GPU deployment). The DiT body's `action_in_proj`
> receives bf16 inputs but `sample_noise` returns fp32, causing:
>
> ```
> RuntimeError: mat1 and mat2 must have the same dtype, but got Float
>   and BFloat16
> ```
>
> ### Reproduce
>
> ```python
> with torch.autocast("cuda", dtype=torch.bfloat16):
>     out = policy.select_action(batch)   # fails inside DiT
> ```
>
> ### Proposed fix
>
> Make both functions accept a `dtype` kwarg, or inspect the autocast
> state and follow it:
>
> ```python
> def sample_noise(shape, device, dtype=None):
>     if dtype is None and torch.is_autocast_enabled():
>         dtype = torch.get_autocast_gpu_dtype()
>     dtype = dtype or torch.float32
>     return torch.normal(
>         mean=torch.zeros(shape, dtype=dtype, device=device),
>         std=torch.ones(shape, dtype=dtype, device=device),
>     )
> ```
>
> Same for `sample_time`. Backwards-compatible — defaults to fp32 when
> not under autocast.
>
> ### Workaround in current downstream code
>
> Monkey-patch at import time:
>
> ```python
> from utils import sampling
> orig = sampling.sample_noise
> def sample_noise_dtype_aware(*a, **kw):
>     out = orig(*a, **kw)
>     if torch.is_autocast_enabled():
>         out = out.to(torch.get_autocast_gpu_dtype())
>     return out
> sampling.sample_noise = sample_noise_dtype_aware
> ```
>
> Works but is fragile (relies on import order, breaks on multi-process).

---

## #3 — docs: clarify `robot_type` field — what strings does the model recognise?

**Title**: `docs: document the robot_type field's accepted values + extension path`

**Body**:

> ### Why
>
> The `robot_type` field in the batch dict is currently undocumented.
> Reading the source (`utils/vlm_utils.py:get_user_prompt`) reveals
> that `robot_type` is **not a learned hardware embedding** — it's a
> string that gets templated into the user prompt that goes to
> Qwen3-VL.
>
> This is an interesting and uncommon design choice (Spirit relies on
> the VLM's language understanding to handle hardware differences,
> instead of using a learned per-robot embedding like some other VLAs).
> But because it's undocumented, downstream cross-embodiment users have
> no way to know:
>
> 1. Which `robot_type` strings has the open-sourced ckpt actually been
>    trained with? The `RoboChallengeDataset` loader currently only
>    supports `Franka` (per `dataset.py:32` and the `move_objects_into_box`
>    constraint), but the README mentions ARX5/aloha/Franka/UR5 as the
>    benchmark coverage. Which of those are in the public ckpt?
> 2. What happens if I pass a string the model has never seen, like
>    `"so100"` or `"my_custom_arm"`? Best practices?
> 3. How would the recommended path look for adding a new robot type?
>    Just edit the prompt template? Or also add training data?
>
> ### Proposed change
>
> Add a `docs/cross_embodiment.md` (or extend the README) covering:
>
> - The list of `robot_type` strings the open-sourced ckpt has actually
>   been trained on
> - How `robot_type` flows through the model (prompt vs embedding)
> - Recommended usage when deploying on a never-seen-before robot
>   (closest match? new prompt template?)
> - When adding a new robot, the recommended workflow

---

## #4 — docs: add a "Running on consumer GPUs" deployment guide

**Title**: `docs: add a "Running on consumer GPUs" guide`

**Body**:

> ### Why
>
> Several non-trivial workarounds are required to run Spirit v1.5 on a
> single consumer GPU (RTX 3090 / 4090 / similar 24GB cards). None of
> them are documented. The set:
>
> 1. The fp32 → bf16 cast workaround (Bug 3 in PR #1 above — addressed
>    when that PR lands)
> 2. Image stays fp32 even when state/action go bf16 (numpy<2.1 doesn't
>    support bf16 conversion in `preprocess_rb_batch`)
> 3. The `sample_noise` dtype fix (Bug 2 above — addressed when that PR
>    lands)
> 4. Final-output `.cpu().numpy()` needs `.float()` first
> 5. `torch.autocast(dtype=torch.bfloat16)` wrap the entire forward
>    (catches a residual fp32 drift inside DiT internals)
>
> Once all 5 are applied, Spirit v1.5 runs at ~6 Hz / 10 GB / RTX 3090.
> But discovering each takes meaningful debugging time.
>
> ### Proposed change
>
> Add `docs/consumer_gpu_deployment.md` covering:
>
> - Hardware sizing matrix (RTX 3090 / 4090 / A6000 / A100 / H100 / etc.)
> - The 5 workarounds above with copy-pasteable code
> - Performance numbers per GPU (latency mean + p99)
> - When NOT to use bf16 (training, accuracy-sensitive eval)

---

## Submission order

| When | Item | Type | Reason |
|---|---|---|---|
| First | #4 docs | issue | Lowest stakes — propose docs and ask if maintainers welcome a PR |
| Same time | #3 docs | issue | Same approach |
| Then | #2 fix | PR | Single-file change, easiest to land |
| Last | #1 feature | PR | Largest change, may need API design discussion |

All submitted under "Liu Zhi (Independent)" — no affiliation.

Open to community feedback. If you've reproduced Spirit on consumer GPUs and have a different workaround stack, file an issue or PR against this doc.
