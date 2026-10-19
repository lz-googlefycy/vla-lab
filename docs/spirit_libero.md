# Spirit v1.5 on LIBERO

Spirit's Franka policy can represent LIBERO's end-effector targets: the first
seven action dimensions are translation, rotation vector, and gripper opening;
the remaining seven dimensions are unused for a single arm. This path uses the
external Spirit policy directly. It does not use the XLeRobot 12-to-14 joint
padding prototype.

The released Spirit base checkpoint is not a published LIBERO checkpoint.
Task/domain adaptation and an actual success-rate evaluation are required.
The [official Spirit instructions](https://github.com/Spirit-AI-Team/spirit-v1.5)
provide a separate checkpoint for RoboChallenge's move-objects-into-box task.
That real-robot result is not a LIBERO baseline.

## Supported workflow

1. Reconstruct official LIBERO demonstration states and render three cameras.
2. Reconstruct OSC controller goals as physical Spirit targets relative to the current state.
3. Fine-tune DiT attention with LoRA and the state/action projection layers.
4. Save the trainable weights together with normalization statistics and action rate.
5. Reload with the same LoRA rank/alpha and run the existing LIBERO evaluator.

The generic DPO pair collector stores a full episode of OSC commands with one
initial image. Those files are not conditioned Spirit action chunks. Spirit
pair collection rejects that format explicitly. The differentiable velocity
score accepts prepared physical `(B, n_action_steps, 14)` targets; its surrogate
quality and an end-to-end DPO/GRPO experiment are not established by this work.

## Camera, state, action, and timing conventions

| Spirit input | LIBERO input |
| --- | --- |
| `cam_high` | `agentview`, global view |
| `cam_left_wrist` | `robot0_eye_in_hand`, wrist view |
| `cam_right_wrist` | `sideview`, lateral view proxy |
| state dimensions 0:3 | EE position in robot-base coordinates, meters |
| state dimensions 3:6 | EE orientation as a base-frame rotation vector, radians |
| state dimension 6 | Measured finger opening, meters |

The side view is a proxy, not a calibrated match to the real RoboChallenge
camera. Training and evaluation use the same mapping. Each view is rendered
at height 240, width 320. MuJoCo/robosuite 1.4 returns OpenGL bottom-up pixels:
the bridge flips rows once to produce upright RGB. It does not apply the
OpenVLA-specific 180-degree image transform.

Images and proprioception are reconstructed from the same recorded MuJoCo
state. The converter relocates legacy asset paths and preserves recorded scene
geometry; it does not silently replace missing views or states with zeros.

For observation index t, reconstruct the servo goal G[t] from recorded state
S[t] and the scaled OSC command a[t]. The first target is this command's goal,
not the measured EE pose one tick later. Translation labels for a chunk are
`G_xyz[t+i] - S_xyz[t]`; rotation labels are
`G_rotation[t+i] * inverse(S_rotation[t])`. The active controller's scaling
and the robot's base rotation are used in the reconstruction.

This distinction is observable in replay: using measured future poses as
stored relative targets failed on the first three Spatial demonstrations
(0/3), while reconstructed controller-goal targets completed all three (3/3).
Both tests replay targets without a learned VLA. Controller tracking lag makes
measured poses different from the commands that generated the trajectories.
Prepared schema 2 stores explicit controller goals and rejects old files.

LIBERO gripper commands map to desired physical openings: -1/open becomes
0.08 m and +1/close becomes 0 m. This avoids treating an object's contact width
as an open command. Targets beyond the episode and the unused arm dimensions
are masked in the loss, including at the final observation.

Prepared data retains LIBERO's recorded 20 Hz control rate. A 60-target chunk
therefore covers three seconds. The evaluator executes the first 10 control
ticks, then observes the scene again. Both action rate and replan interval are
explicit; a pretrained checkpoint's action rate must not be inferred from the
number 60. If requested rates differ, interpolation includes the current pose
at t=0 and samples targets starting at t=1/control_hz.

Every target remains relative to the chunk-start observation. Before each
`env.step`, the bridge computes the error from the live pose and inverts the
active controller's input/output scaling. This avoids repeatedly accumulating
the same chunk-relative displacement. Success comes from `check_success()`;
an environment termination alone is not scored as success.

## Setup

Use a separate environment with Spirit and LIBERO installed. Export:

```bash
export SPIRIT_SRC=/path/to/spirit-v1.5
export SPIRIT_CKPT=/path/to/Spirit-v1.5
export QWEN_ASSETS=/path/to/Qwen3-VL-4B-Instruct-assets
export PYTHONPATH="$PWD/code:$SPIRIT_SRC"
pip install h5py wandb
```

`QWEN_ASSETS` supplies config, tokenizer, and image processor assets. Qwen
weights are populated by Spirit's complete floating-point `model.safetensors`
with strict key checking. An AWQ asset directory can be used: its
`quantization_config` is removed, and its packed integer weights are not loaded.
No meta tensors or checkpoint config-file rewrites are used. Embedding/LM-head
weight sharing is retained even when temporary random initialization is skipped.

Configure LIBERO paths following its own setup instructions. If using
`LIBERO_CONFIG_PATH`, point it to the directory containing `config.yaml`.
Tested environment: torch 2.8.0+cu128, transformers 4.57.6, diffusers 0.37.1,
robosuite 1.4.0, mujoco 3.2.7, numpy 2.3.5, scipy 1.18.0, h5py 3.16.0.

## Prepare, train, evaluate

Download an official HDF5 from
[LIBERO-datasets](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).
For a small pipeline check:

```bash
python -m spirit_adapter.libero_dataset \
  --source /path/to/task_demo.hdf5 \
  --output /path/to/spirit_task.hdf5 --max_episodes 3

python -m spirit_adapter.train_libero \
  --base_ckpt "$SPIRIT_CKPT" --backbone "$QWEN_ASSETS" \
  --dataset /path/to/spirit_task.hdf5 --episodes demo_0 \
  --output_dir output/spirit_libero_sft --max_steps 1 \
  --lora_r 4 --lora_alpha 8

python -m post_training.eval_libero \
  --base spirit --base_ckpt "$SPIRIT_CKPT" \
  --spirit_backbone "$QWEN_ASSETS" \
  --lora_ckpt output/spirit_libero_sft/adapter.pt \
  --lora_r 4 --lora_alpha 8 \
  --suites libero_spatial --n_tasks 1 --n_trials_per_task 1 --seeds 42 \
  --spirit_replan_steps 10 --capture_videos 1 \
  --output_dir output/spirit_libero_eval
```

For CPU checks, train with `--device cpu --dtype float32`, and evaluate with
`--spirit_device cpu --base_dtype float32`. Do not hide all CUDA devices to
select a CPU policy: the EGL renderer can still use a GPU independently.
The saved adapter restores the training action rate and all three normalization
modules; norm arithmetic remains float32 when neural weights are bfloat16.

A one-step run tests loading, gradients, serialization, and control plumbing.
It is not a trained benchmark model. For training, prepare more demonstrations,
reserve held-out episodes, and increase steps. Report policy success over
explicit task/initial-state/seed counts, separately from training loss and
controller replay checks.

## Validation

```bash
python -m unittest spirit_adapter.test_libero_bridge -v

python -m spirit_adapter.validate_libero_bridge \
  --source /path/to/task_demo.hdf5 --episodes 3 \
  --prepared /path/to/spirit_task.hdf5 \
  --output output/bridge_validation.json
```

The second command replays targets from the actual converted training file
through the live bridge. It runs no VLA. On the first three demonstrations of
LIBERO Spatial task 0, original and prepared-target replays both succeed
(3/3). Maximum EE path differences from original command replay are 3.3, 7.7,
and 3.5 mm. Full simulator states are not numerically identical. The separate
arm-command inversion check has maximum absolute error 1.3e-15.

The [validation artifacts](../assets/spirit_libero/) include the source data
hash, source revisions, prepared-target replay, and measured-target ablation.
These counts validate data/control compatibility, not Spirit policy success.

A real Spirit base-weight CPU SFT step completed with strict checkpoint keys
(0 missing, 0 unexpected) and finite loss/gradients. The saved trainable weights
and normalization statistics were reloaded for a full closed-loop trial. That
one-step pipeline checkpoint scored 0/1 and is not a trained benchmark model.
The single-task experiments below provide separate policy evidence; the
one-step check alone establishes only pipeline execution.

### Single-task 500-step pilot (2026-10-19)

A native FP32 evaluation of the 500-step LoRA checkpoint scored **0/10**
on LIBERO Spatial task 0, using official initial-state indices 0 through 9
and seed 42. Every trial reached the 220-step limit with no execution error.
This pilot trained on demo_0 through demo_39 only (batch size 1); demo_40
through demo_49 were excluded from training and normalization statistics.
The official evaluation initial states have not been established to match
those held-out demonstrations, so this is not a held-out-demonstration score.
See [the policy evaluation record](../assets/spirit_libero/policy_step500_eval.json)
for checkpoint hash, settings, and all trial outcomes. Ten rollout videos are
stored in the local evaluation output directory recorded there.

This result does not establish a successful learned-policy adaptation.
Training loss and the 3/3 expert-target replay remain separate evidence;
policy predictions and trajectories require diagnosis before making a
performance claim.

### Diagnosing the 500-step pilot

A paired offline check sampled 24 training and 24 held-out observations
(6 episodes per split, four phases per episode). With common initial noise,
the held-out mean translation error over the first 10 targets decreased from
151.1 mm for Base with training-only LIBERO normalization, to 83.3 mm at
step 100 and 65.8 mm at step 500. The corresponding gripper-command accuracy
at step 500 was 61.25%. These are sampled teacher-observation action errors,
not closed-loop success rates. The original batch-1 pilot consumed only
500 of the 4130 training observations before completing one shuffled pass.
See [the offline diagnosis](../assets/spirit_libero/policy_step500_diagnosis.json).

Four reconstructed demonstration states reproduced their prepared image
pixels, low-dimensional states, and controller goals exactly. This checks
those samples, rather than every observation in the dataset.

For a controlled demonstration replay, restore the recorded XML and state
together through `run_one_trial(..., reset_callback=...)`, for example a
callback invoking `reset_demo(env, original_demo)`. Use `num_steps_wait=0`
when replaying an already settled recorded state. A second default hard reset
discards the recorded XML, resamples fixed scene elements, and prevents an
exact training-input comparison. Official benchmark trials continue to use
the default reset and settling protocol.

A separate one-demonstration diagnostic trained on demo_0 alone (batch 8,
1000 updates followed by another 3000 updates with a fresh optimizer). At
4000 cumulative updates it scored 1/3 seeds on the exact seen demonstration
scene and 0/3 on the exact held-out demo_40 scene. It also scored 1/10 on the
standard task-0 benchmark initial states 0..9, seed 42. Thus a learned
policy can complete the task through the bridge, but behavior remains
unreliable. Its data and normalization differ from the 40-demo pilot;
this is not a controlled measurement of the effect of update count alone.
See [the overfit diagnostic record](../assets/spirit_libero/overfit_diagnosis.json).

Expanding the prepared-goal replay to all 50 demonstrations completed 49/50
converted-target replays. Original-command replays completed 48/50
(demo_34 and demo_37 failed); converted-target replay also failed on demo_37.
The validator therefore exited 1 under its requirement that both replays
succeed. These failures remain unresolved and recorded rather than being
counted as passing validation. Maximum EE path difference was 58.2 mm over
all episodes. This is a simulator replay check, not learned-policy success.
See [all 50 replay records](../assets/spirit_libero/prepared_goal_replay_all50.json).

### Forty-demonstration continuation (2026-10-19)

The original step-500 adapter was continued for 4000 additional updates with
batch size 8 (FP32, LoRA rank 32/alpha 64, AdamW at fixed 5e-5). A fresh
optimizer was initialized from adapter weights; this is a weight-only
continuation. Training used demo_0..39, and normalization statistics were
computed only from those episodes. Demo_40..49 remained excluded. The
continuation consumed 31958 observations across shuffled passes.

Frozen Qwen features were cached without changing the training objective.
A comparison against the official policy forward gave cached loss
0.1083897427 versus 0.1083899364. No model weights were quantized.

| Checkpoint | Official task-0 initial states 0..9, seed 42 |
| --- | --- |
| Original 500 updates, batch 1 | 0/10 |
| +1000 updates, batch 8 | 0/10 |
| +4000 updates, batch 8 | **9/10** |

The final checkpoint separately scored **7/10** on recorded initial states
from held-out demo_40..49, seed 42. Those trials restored each recorded XML
and state together, skipped settling, and verified that initial cameras and
state exactly matched the prepared data. Each initial success predicate was
false. They use a different initialization protocol from the official
benchmark and must be reported separately. Both protocols capped episodes at
220 steps, predicted at 20 Hz, replanned every 10 control ticks, and used
`check_success()`. No inference or environment execution errors occurred.
The official trial with init index 6 failed; held-out demos 41, 45, 47 failed.

On the same 48 teacher observations and paired noise used for the earlier
offline diagnosis, held-out first-10-target mean position error decreased
from 65.8 to 41.2 mm, and gripper-command accuracy increased from 61.25% to
87.08%. These offline errors are separate from closed-loop success.

These results support training budget as a substantial factor in the
original failure, and demonstrate a working learned-policy control path.
They cover one task and ten trials per protocol, not all Spatial tasks or
a full-suite result. Full trial logs, settings, hashes, and input checks are
in [the continuation evaluation record](../assets/spirit_libero/policy_continuation_eval.json).

### Expanded fixed-checkpoint evaluation (2026-10-19)

The same final FP32 adapter was evaluated without further training over 50
official task-0 initial states and seeds 42, 1337, 2026 (150 trials), plus the
10 held-out demonstration starts under the same three seeds (30 trials).
These protocols remain separate. All 180 prescribed trials completed with
no inference or environment execution error; every unsuccessful rollout
reached the 220-step cap. All held-out initial images and states matched
the prepared data exactly, and no trial began with success already true.

| Official initial-state indices | Seed 42 | Seed 1337 | Seed 2026 |
| --- | --- | --- | --- |
| 00-09 | 10/10 | 4/10 | 0/10 |
| 10-19 | 9/10 | 3/10 | 3/10 |
| 20-29 | 10/10 | 5/10 | 0/10 |
| 30-39 | 10/10 | 3/10 | 0/10 |
| 40-49 | 8/10 | 1/10 | 1/10 |
| Total | 47/50 | 16/50 | 4/50 |

| Protocol | Seed 42 | Seed 1337 | Seed 2026 | Combined |
| --- | --- | --- | --- | --- |
| Official starts | 47/50 | 16/50 | 4/50 | **67/150 (44.7%)** |
| Held-out demonstration starts | 7/10 | 3/10 | 2/10 | **12/30 (40.0%)** |

The seed is reset per trial, matching the repository evaluator: a given
seed shares its action-noise stream across initial states. These counts
therefore describe the tested three-seed protocol rather than independent
noise samples for every initial state.

Because all GPUs were occupied by other workloads, Qwen weights were
offloaded through CPU while preserving FP32 execution. The conditional
generation forward and its hidden-state capture were retained; an unused LM
head was omitted only after exact native CPU action-output parity. A GPU
probe matched ordinary staged native execution exactly; first-batch checks
for batch4 differed by at most 2.1e-6 in active action channels. The first
16 completed trials used batch2, and the remaining 164 used batch4. These
checks establish numerical agreement on sampled inputs, not bitwise
closed-loop replay across different runtime configurations. The previous
seed42 init6 failure became a success in the batched repeat; both records
are retained.

The expanded results demonstrate a working single-task control path but
**strong sensitivity to the evaluation seed**. The earlier 9/10 at seed42
should not be presented as robust multi-seed performance. The other nine
Spatial tasks and the other suites have not been evaluated by this run.
See [all expanded evaluation records](../assets/spirit_libero/policy_expanded_eval.json)
for every outcome, checkpoint hash, numerical probes, and runtime settings.

### Small cross-task transfer evaluation (2026-10-19)

The same task-0-trained FP32 checkpoint was evaluated on four additional
Spatial tasks without further training. Each task used official initial
states 0, 1, 2 and seeds 42, 1337, 2026: nine rollouts per task, 36 total.
No demonstration from these four tasks was used for training or normalization.

| Spatial task index | Instruction summary (pick bowl and place on plate) | Success |
| --- | --- | --- |
| 1 | Bowl next to the ramekin | 0/9 |
| 2 | Bowl at table center | 0/9 |
| 3 | Bowl on the cookie box | 0/9 |
| 4 | Bowl in the cabinet's top drawer | 0/9 |
| Total | Four tasks, small transfer sample | **0/36 (0%)** |

All prescribed trials completed without inference or environment execution
errors. No rollout began with success already satisfied; every failure
reached the 220-step limit. Settings remained 20 Hz, replanning every ten
control ticks, ten settling ticks, and explicit `check_success()` evaluation.
Native conditional-generation hidden-state capture and FP32 weights were
preserved with Qwen CPU offload. Each worker batched only one task; initial
batch4 versus single-input active-action differences were at most 5.7e-6.
Each seed was reset per rollout, as in the preceding evaluation.

This sample provides no evidence of successful cross-task transfer from
the single-task adapter. It does not evaluate the untouched Spirit base
policy, training on those four tasks, the full Spatial suite, or any
exported/quantized model. A zero score alone cannot distinguish poor policy
generalization from a task-specific interface issue. A suitable next check
is replaying controller targets on the additional tasks, followed by a
training/held-out evaluation on multiple tasks if those checks pass.

Full outcomes, exact instructions, checkpoint/source hashes, manifests,
and the previous task-0 reference are in
[the cross-task evaluation record](../assets/spirit_libero/policy_cross_task_eval.json).
Local runners, launch commands, videos, and a first-trial contact sheet are
under `output/spirit_libero/cross_task_eval_20261019/`.
