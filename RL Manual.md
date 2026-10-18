# gen0WaLTER

![License](https://img.shields.io/badge/License-INSERT__LICENSE-blue)
![Build Status](https://img.shields.io/badge/Build-INSERT__STATUS-lightgrey)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)

**gen0WaLTER** trains a PPO policy for the full-scale WaLTER robot on flat ground. The task is body-frame forward speed and yaw rate. The student acts with the same 16-number command the classical full-body PD controller uses: four hips, four absolute knee angles, and eight wheel velocities. `head_joint` is fixed at 0 degrees and is not an action.

A second mode treats `FullBodyPdController` as a teacher. Before every physics step the environment records the action the PD controller would have sent from that state. The student is rewarded for matching it. The match term is shaped so its peak stays below forward tracking plus yaw tracking, so the teacher steers the policy and the velocity task still ranks higher.

Playback covers both kinds of controller. `scripts/play.py --policy pd` runs the classical PD policy, including the existing terrain scenes. `scripts/play.py --policy checkpoint` loads an rsl-rl checkpoint from either the plain velocity run or the teacher run. Physics, the MJCF, and the PD gain file are documented in [`SIMULATION.md`](SIMULATION.md) and [`README.md`](README.md).

## Key Features

- Flat velocity task with forward speed in `[0.25, 1.25]` m/s, averaging 1 m/s, and `ang_vel_z` in `[-1, 1]` rad/s, zero on half of resamples. Lateral speed stays at zero.
- PPO through rsl-rl 5.4 (`OnPolicyRunner`) on the vendored mjlab `ManagerBasedRlEnv`.
- Separate actor and critic MLPs, observation normalization, and a Gaussian policy.
- Teacher-student training: the PD action is snapshotted before physics, then scored with an exponential match reward of weight `0.5`.
- Weights & Biases project `walter-rl` for scalars, the training config, and checkpoints. TensorBoard files are still written under `logs/rsl_rl/`.
- Checkpoint playback in the Viser browser viewer or MuJoCo's native viewer, and the existing PD play scripts stay available.

# User Manual

## Prerequisites

| Requirement | Version / notes |
| --- | --- |
| Python | `>=3.12,<3.14` (the checked-in environment uses **3.13**) |
| [uv](https://docs.astral.sh/uv/) | **0.11.19** or newer (`uv_build>=0.12.1,<0.13`) |
| Operating system | Linux. mjlab sets `MUJOCO_GL=egl` on Linux before MuJoCo imports. |
| GPU | Optional NVIDIA GPU. Training uses `cuda:0` when PyTorch can see it, and `cpu` otherwise. |
| Weights & Biases account | A logged-in `wandb` user. The project name is `walter-rl`. |
| Viewer | A browser for `--viewer viser`, or a local display for `--viewer native`. Playback only. |

RL-specific packages from `pyproject.toml`:

| Package | Pin |
| --- | --- |
| rsl-rl-lib | `5.4.2` |
| wandb | `>=0.22.3` (the lockfile resolves **0.30.0**) |
| tensorboard | `>=2.20.0` |
| tensordict | unpinned; required by the rsl-rl observation dict |
| mjlab | editable path source, `mjlab/` |

The simulation stack (torch, MuJoCo, Warp, Viser) comes in through mjlab. See [`README.md`](README.md) for those pins.

## Installation

Clone the repository and sync the lockfile. `walter_rl` depends on the vendored `mjlab` tree.

```bash
git clone [INSERT REPOSITORY URL HERE]
cd gen0WaLTER
uv python install 3.13
uv sync
wandb login
```

`uv sync` creates `.venv` and installs `walter_rl`, `mjlab`, `rsl-rl-lib`, and `wandb`. `wandb login` writes credentials to `~/.netrc`. Training reads that file on the next run. Set `WANDB_USERNAME` when the run should be filed under an entity other than the logged-in user.

## Quick Start / Usage

Smoke-test the velocity task. The script defaults (`--num-envs 8`, `--max-iterations 2`) exist for this check. They are too short for a policy you would play back.

```bash
uv run python scripts/train.py --num-envs 8 --max-iterations 2
```

A training run of the flat task:

```bash
uv run python scripts/train.py --num-envs 4096 --max-iterations 5000
```

The same length with the PD teacher:

```bash
uv run python scripts/train.py --teacher --num-envs 4096 --max-iterations 5000
```

Teacher training on the rough-test scene:

```bash
uv run python scripts/train.py --teacher --task rough-test --num-envs 4096 --max-iterations 500
```

Continue from a checkpoint. `--max-iterations` counts updates after that file, and the new checkpoints go in a fresh timestamp directory:

```bash
uv run python scripts/train.py \
  --resume logs/rsl_rl/walter_fs_velocity/<timestamp>/model_500.pt \
  --max-iterations 500
```

Open the browser control panel to play a policy or start and continue training:

```bash
uv run python scripts/panel.py
```

Then open <http://127.0.0.1:8765>. Playback still uses the Viser viewer or the native window. The panel links to the viewer once it is up.

Lower `--num-envs` when the GPU runs out of memory. One learning iteration collects 24 control steps from every environment. The control step is 0.02 s, so one iteration is 0.48 s of simulated time per environment. Five thousand iterations is 40 minutes of experience per environment.

Play a checkpoint. The path is printed as `Logs:` at the start of training, and again in the Weights & Biases run.

```bash
uv run python scripts/play.py --policy checkpoint \
  --checkpoint logs/rsl_rl/walter_fs_velocity/<timestamp>/model_5000.pt
```

Play the classical PD policy from the same script:

```bash
uv run python scripts/play.py --policy pd --task flat
uv run python scripts/play.py --policy pd --task rough --viewer native
uv run python scripts/play.py --policy pd --task flat --payload rider
```

`scripts/play_pd.py` and `scripts/play_smooth_tumble.py` are unchanged. Use those when you want the PD-only viewers described in [`README.md`](README.md).

Confirm the teacher reward before a long teacher run:

```bash
uv run python scripts/check_teacher_reward.py
```

A passing run prints unscaled reward rates for the teacher action and for a zero action, then `Teacher tracking reward passed.`

## Configuration

There is no `.env` file. Training behavior comes from CLI flags, `src/walter_rl/rl/ppo.py`, and `walter_fs_rl_env_cfg` / `walter_fs_teacher_env_cfg` in `src/walter_rl/tasks/velocity/config/walter_fs/env_cfgs.py`.

### Training CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--num-envs` | `8` | Parallel environments. Use thousands for a real run. |
| `--max-iterations` | `2` | PPO updates. Use `5000` for a full run. |
| `--task` | `flat` | `flat`, `rough`, `rough-test`, `stairs`. Same scenes as playback. |
| `--teacher` | off | Train in `TeacherStudentEnv` and add the PD match reward. |
| `--device` | GPU if present | `cuda:0` or `cpu`. |
| `--resume` | none | Checkpoint to continue from. `--max-iterations` counts additional updates. Logs go to a new timestamp directory and a new W&B run. |

The seed is `42` in both the environment and the runner. It is not a CLI flag.

### What one run writes

| Output | Location |
| --- | --- |
| Checkpoints and TensorBoard | `logs/rsl_rl/walter_fs_velocity/<YYYY-MM-DD_HH-MM-SS>/` |
| Teacher run | `logs/rsl_rl/walter_fs_teacher/<timestamp>/` |
| Other scenes | Same folders with a `_<task>` suffix, for example `walter_fs_teacher_rough-test` |
| Weights & Biases | Project [`walter-rl`](https://wandb.ai). Run name is the timestamp directory. |
| Local W&B files | `wandb/run-<timestamp>-<id>/` at the repo root |

Checkpoints are saved every 50 iterations (`save_interval` in `walter_ppo_runner_cfg`) and at the end of the run. `model_<iteration>.pt` is the file `scripts/play.py` loads. The same files are uploaded to the W&B run.

### Commands and rewards

The RL task starts from the flat scene and replaces the terrain-training command and reward set.

| Command | Range |
| --- | --- |
| `lin_vel_x` | `(0.25, 1.25)` m/s, cube-root bias, average 1 m/s |
| `lin_vel_y` | locked at `0` |
| `ang_vel_z` | `(-1.0, 1.0)` rad/s, and `0` on half of resamples |
| heading, ride height | omitted. The command vector is 3 numbers. |

Five percent of resamples are a full stop (`rel_standing_envs = 0.05`), chosen independently of the yaw coin flip. Episodes last 20 s during training. Playback sets the episode length to a very large number so the viewer keeps stepping.

| Reward | Weight | Role |
| --- | --- | --- |
| `track_linear_velocity` | `2.0` | Body-frame XY speed. Standard deviation `0.5`. |
| `track_angular_velocity` | `1.0` | Yaw rate. Standard deviation `sqrt(0.5)`. |
| `upright` | `0.1` | Torso gravity alignment. |
| `action_rate_l2` | `-0.02` | Penalize action changes. |
| `head_level` | `-0.1` | Penalize waist motion. |
| `wheel_wheel_collision` | `-1.0` | One count for each pair of wheels that are touching. |
| `knee_stuck_tumble` | `0.25` | PD stall integral times each knee's forward rate. Pays for a front-down tumble only while the body is behind a forward command. |
| `track_teacher` | `0.5` | Teacher runs only. Kernel is in `[0, 1]`, so the term contributes at most `0.5`. |

The teacher kernel is `exp(-mean((a_student - a_teacher)^2) / std^2)` with `std = 1.0`. `Episode_Metrics/teacher_action_mse` is the mean squared action gap and is logged, not added into the return. Forward tracking peaks at `2.0` and yaw tracking at `1.0`, so a perfect teacher match cannot outrank the velocity task.

### Action and observation

The action is 16 numbers, in this order:

| Slice | Size | Meaning |
| --- | --- | --- |
| `hip_pos` | 4 | Hip angle, scaled by the 70° hip limit and clipped to that limit. |
| `knee_pos` | 4 | Absolute knee angle in radians, about the default pose. |
| `wheel_vel` | 8 | Wheel rate. The action is multiplied by `30` rad/s. |

`head_joint` is constrained to 0 degrees. Actor and critic both see a 383-number vector: base linear and angular velocity, projected gravity, hip and waist position, all joint velocities, the previous action, the 3-number command, a 325-ray terrain height scan, and knee sin/cos. The scan is a 2.4 m by 1.2 m grid at 0.1 m spacing, shifted 0.4 m forward on the torso, and each height is divided by the 5 m ray limit. The actor's joint positions and the scan include noise during training. Wheel and knee angles are omitted as raw positions because those joints are continuous.

### PPO

Settings live in `walter_ppo_runner_cfg`:

| Setting | Value |
| --- | --- |
| Networks | Actor and critic MLPs `(512, 256, 128)`, ELU, observation normalization |
| Policy | Gaussian, scalar standard deviation, initial std `1.0` |
| Rollout | 64 steps per environment, 5 epochs, 4 mini-batches |
| Optimizer | Adam, learning rate `1e-3`, adaptive schedule, desired KL `0.01` |
| Returns | discount `0.99`, GAE lambda `0.95`, entropy coefficient `0.005` |
| Clip | `0.2`, value loss coefficient `1.0`, gradient norm `1.0` |

### Playback CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--policy` | `pd` | `pd` or `checkpoint` |
| `--checkpoint` | none | Required for `--policy checkpoint` |
| `--task` | `flat` | `flat`, `rough`, `rough-test`, `stairs`. Terrain for both policies. |
| `--num-envs` | `1` | Parallel robots in the viewer |
| `--viewer` | `viser` | `viser` or `native` |
| `--payload` | none | Rigid mass welded to the torso. Used by `--policy pd`. `rider` is a 75 kg seated human. |

Checkpoint playback uses `--task` for the ground and the same 384-number observation on every scene, including teacher checkpoints. `--payload` applies when `--policy pd`. Checkpoints trained before the terrain scan was added are 59-number actors and do not load.

### Environment variables

| Variable | Effect |
| --- | --- |
| `WANDB_USERNAME` | W&B entity. Unset uses the logged-in user. |
| `WANDB_API_KEY` | Optional alternative to `wandb login` for a headless machine. |

# File Explanations (Directory Structure)

```
gen0WaLTER/
├── pyproject.toml
├── uv.lock
├── README.md
├── SIMULATION.md
├── RL Manual.md
├── scripts/
│   ├── train.py
│   ├── play.py
│   ├── panel.py
│   ├── check_teacher_reward.py
│   ├── play_pd.py
│   └── play_smooth_tumble.py
├── src/walter_rl/
│   ├── panel/
│   ├── rl/
│   │   ├── ppo.py
│   │   ├── vecenv.py
│   │   ├── teacher_env.py
│   │   ├── wandb_logger.py
│   │   └── checkpoint_policy.py
│   ├── tasks/velocity/
│   │   ├── mdp/rewards.py
│   │   └── config/walter_fs/env_cfgs.py
│   └── control/full_body_pd/
│       ├── controller.py
│       ├── env.py
│       └── policy.py
├── mjlab/
├── logs/rsl_rl/
└── wandb/
```

- `scripts/train.py` builds the scene from `--task`, wraps it for rsl-rl, and calls `OnPolicyRunner.learn`. `--teacher` selects `TeacherStudentEnv` and the `walter_fs_teacher` log folder. Other scenes use a `_<task>` suffix. `--resume` loads a checkpoint's weights, optimizer, and iteration counter before `learn`.
- `scripts/panel.py` serves the browser control panel on `127.0.0.1:8765`. The page starts `play.py`, `play_smooth_tumble.py`, or `train.py`, and it can stop that process.
- `scripts/play.py` opens the viewer on `FullBodyPdPolicy` or on a loaded actor.
- `scripts/check_teacher_reward.py` rolls out a fixed forward and yaw command, once following the teacher and once with zeros, and checks the match term.
- `src/walter_rl/rl/ppo.py` is the runner dict: networks, PPO hyperparameters, the W&B logger class, and the checkpoint interval.
- `src/walter_rl/rl/vecenv.py` adapts `ManagerBasedRlEnv` to the rsl-rl 5.4 vector env. It resets once at construction because the runner does not.
- `src/walter_rl/rl/teacher_env.py` calls `FullBodyPdController.act()` before `step` and stores `teacher_action`. `follow_teacher` applies that action and is used by the check script.
- `src/walter_rl/rl/wandb_logger.py` starts the `walter-rl` run and stores a JSON-safe copy of the configs. Reward functions are saved as strings.
- `src/walter_rl/rl/checkpoint_policy.py` turns the viewer observation dict into the TensorDict the actor expects.
- `env_cfgs.py` defines `walter_fs_rl_env_cfg` and `walter_fs_teacher_env_cfg`, including the reward weights.
- `rewards.py` holds `track_teacher_action` and `teacher_action_mse`.
- `control/full_body_pd/` is the teacher. `_add_waist_lock` gives the RL task the same waist actuator, absolute knees, and hip and knee gains as the PD controller, and it copies the shared articulation before editing it.
- `mjlab/` is the vendored environment library. This repo trains with rsl-rl directly and does not use an upstream mjlab `train.py`.
- `logs/rsl_rl/` and `wandb/` are run outputs. Leave them out of commits.

# Developer Notes

## System Architecture

Training is a thin adapter around two libraries. mjlab owns the scene, commands, observations, rewards, and the physics step. rsl-rl 5.4 owns PPO. `RslRlVecEnvWrapper` is the boundary: it exposes actor and critic observation groups and returns the step tuple `OnPolicyRunner` expects.

`walter_fs_rl_env_cfg` starts from the flat velocity scene, then installs the simple command ranges and the reward terms above. It also calls `_add_waist_lock` so the student's action lines up with `FullBodyPdController`. The PD play scenes keep their own terrain, tumble, and ride-height rewards. Those terms are removed only inside the RL config.

Teacher training subclasses the environment rather than calling the controller from the reward. Reward evaluation happens after `sim.step`, so a teacher queried there would see the next state. `TeacherStudentEnv.step` records the PD action first, then applies the student action (or the teacher action when `follow_teacher` is set).

Logging subclasses rsl-rl's `WandbLogWriter`. The installed writer passes `start_method="thread"` into `wandb.Settings`, which Weights & Biases 0.30 rejects, and it tries to upload reward callables. `WalterWandbLogWriter` starts the run without that setting and stringifies values W&B cannot store. The project name is the constant `WANDB_PROJECT` in `wandb_logger.py`.

## Development Setup

Install as in [Installation](#installation), then run a one-iteration job to confirm CUDA and W&B:

```bash
uv run python scripts/train.py --num-envs 4 --max-iterations 1
```

A healthy smoke run prints a W&B run URL, finite `Mean value loss`, `Mean surrogate loss`, and `Mean entropy loss`, and leaves `model_0.pt` in the timestamped log directory. Add `--teacher` to exercise the match reward. The log should include `Episode_Reward/track_teacher` and `Episode_Metrics/teacher_action_mse` once an episode ends. One iteration is often shorter than the 20 s episode, so those episode averages can be absent until a longer run.

`--task` selects the scene. Edit `_configure_simple_commands` and `_configure_simple_rewards` to change the velocity task itself. To change PPO, edit `walter_ppo_runner_cfg`. Keep `rnd_cfg` and `symmetry_cfg` present and set to `None`. The rsl-rl 5.4 runner reads both keys every iteration.

## Testing

There is no pytest suite. The checks that cover the RL path are:

```bash
uv run python scripts/check_teacher_reward.py
uv run python scripts/train.py --num-envs 8 --max-iterations 2
uv run python scripts/train.py --teacher --num-envs 8 --max-iterations 2
```

The reward check fails the process if the match weight is at least `2.0 + 1.0`, if the teacher-following kernel drops below `0.95`, or if zeros score the match term as high as the teacher action. Checkpoint playback can be checked by loading `model_1.pt` from either smoke run with `scripts/play.py --policy checkpoint`.

## Contributing & Future Work

Match the style already in `src/walter_rl/rl/`: one responsibility per function, a docstring on each public function, and comments only where the ordering or a library quirk is doing real work. The teacher snapshot has to stay before the physics step. The match weight has to stay below the forward and yaw weights.

The velocity task is forward speed and yaw on the scene selected by `--task`. The default scene is flat. Ride-height rewards still belong to the PD scenes. The teacher action and the student action have to stay the same length on every scene.

## License

No license file is checked in. [INSERT LICENSE HERE]
