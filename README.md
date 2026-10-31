# gen0WaLTER

![Build Status](https://img.shields.io/badge/Build-In_Progress-lightgrey)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)

**gen0WaLTER** is a simulation package for the full-scale WaLTER robot. The package version is **0.1.0**. WaLTER is a wheeled-leg platform: four hip joints, four continuous knee joints, eight driven wheels, and a waist (`head_joint`) held at zero.

The default learned policy is a state machine on top of the classical full-body PD controller. PPO outputs two numbers on `[-1, 1]` (Beta actor, clipped). A positive first value latches tumbling: every knee spins about +Y and every wheel holds `tumble.wheel_speed`. A non-positive value leaves the PD speed, yaw, pitch, and roll loops in charge. Each tumble edge is held for 0.5 s. The second value maps to a Tank Mode hip offset from 0° to 30° via `(a + 1) / 2` (network −1 → 0°, 0 → 15°, +1 → 30°). Tank Mode stays on and the tank knee command stays at 0°. The stall-integral gate stays out of this loop; `StateMachineEnv` writes the latch with `FullBodyPdController.set_tumble_command`.

`--teacher` is the other training path. The student outputs 17 numbers (hips, wheels, drive state, and knee targets). `TeacherStudentEnv` stores the PD plant action from the state the student saw, then steps the student action.

Playback covers the classical PD controller, the smooth-tumble variant, and a state-machine checkpoint. A local browser panel can start one play or train process. How an MJCF file becomes a joint torque is in [`SIMULATION.md`](SIMULATION.md). Masses and link lengths are in [`src/walter_rl/assets/walter_fs/README.md`](src/walter_rl/assets/walter_fs/README.md).

## Key Features

- State-machine PPO at 10 Hz: tumble toggle plus 0–30° tank hip, Beta actor on `[-1, 1]`, actor MLP 256/128/64, critic MLP 512/256/128. The PD controller still writes the 16 joint targets each physics substep.
- Front and rear height grids (1 m × 1 m, 64 rays each), IMU terms, knee encoding, hip and knee torque, wheel torque, the twist command, last policy action, and 0.5 s of applied tumble memory.
- State-machine rewards (`SM_*` in `env_cfgs.py`): forward tracking (1.0), forward progress (2.0), yaw tracking (1.0), climb tumble (1.5), unneeded tumble (−1.0), downhill tumble (−6.0 per radian nose-down while latched), uprightness (0.1), action rate (−0.1), torso contact (−2.0), limb torque (−0.005), stuck speed (−0.5).
- Optional 17-D teacher-student run (`--teacher`) at 50 Hz with the same scan layout and MLP widths (Gaussian actor). PD action matching uses `TRACK_TEACHER_WEIGHT = 0.5` (peak contribution 0.5, below forward plus yaw tracking at 3.0).
- Classical full-body PD and a smooth-tumble variant that holds hip height level while the knees spin.
- Four scenes: `flat`, `rough`, `rough-test`, and `stairs`.
- rsl-rl 5.4 `OnPolicyRunner`, TensorBoard logs under `logs/rsl_rl/`, and a localhost panel for one play or train job.
- Checked-in full-scale MJCF (`walter_fs.xml`), with generator motors stripped and CubeMars DC motor actuators attached.
- Viser joystick for forward speed, yaw rate, hip angle, knee angle, Tank Mode, and the tumble integral. Checkpoint playback draws the tumble decision on the motors tab.

# User Manual

## Prerequisites

| Requirement | Version / notes |
| --- | --- |
| Python | `>=3.12,<3.14` (the checked-in environment uses **3.13.13**) |
| [uv](https://docs.astral.sh/uv/) | **0.11.19** or newer (`uv_build>=0.12.1,<0.13`) |
| Operating system | Linux is the supported path. mjlab sets `MUJOCO_GL=egl` on Linux before MuJoCo imports. |
| GPU | Optional NVIDIA GPU. Training and `make_pd_env` select `cuda:0` when `torch.cuda.is_available()`, otherwise `cpu`. |
| Viewer | A browser for `--viewer viser`, or a local display for `--viewer native`. |

Pinned simulation stack (from `mjlab/pyproject.toml`):

| Package | Pin |
| --- | --- |
| torch | `2.14.0` |
| mujoco | `~=3.11.0` |
| mujoco-warp | `~=3.11.0` |
| warp-lang | `>=1.14.0` |
| viser | `>=1.0.27` |
| mjviser | `>=0.0.14` |

RL packages from the root `pyproject.toml`:

| Package | Pin |
| --- | --- |
| rsl-rl-lib | `5.4.2` |
| tensorboard | `>=2.20.0` |
| tensordict | unpinned; the rsl-rl adapter packs observations as a `TensorDict` |

## Installation

Clone the repository and sync the lockfile. `walter_rl` depends on the vendored `mjlab` tree as an editable path source.

```bash
git clone [INSERT REPOSITORY URL HERE]
cd gen0WaLTER
uv python install 3.13
uv sync
```

`uv sync` creates `.venv` and installs `walter_rl`, `mjlab`, `rsl-rl-lib`, and `tensorboard` in editable mode for the local packages. CUDA and CPU PyTorch indexes are declared as optional extras on `mjlab` (`cu128` and `cpu`). The root project does not select either extra; the lockfile resolves the default `torch==2.14.0` wheel.

## Quick Start / Usage

Play the PD controller on flat ground. Viser prints a local URL when the server starts; open that URL in a browser.

```bash
uv run python scripts/play.py --policy pd --task flat
uv run python scripts/play_pd.py --task flat
```

`scripts/play_pd.py` is the same PD scene. `scripts/play.py` also loads a trained checkpoint (state machine or teacher; teacher paths under `walter_fs_teacher` are detected automatically):

```bash
uv run python scripts/play.py --policy checkpoint --task rough-test \
  --checkpoint logs/rsl_rl/walter_fs_state_machine_rough-test/<run>/model_500.pt
```

Smooth tumble uses its own script. The hips hold a flat-ground contact height while the knees spin or settle.

```bash
uv run python scripts/play_smooth_tumble.py --task rough --viewer viser
```

Train the tumble toggle and the tank hip offset. The script defaults (`--num-envs 8`, `--max-iterations 2`) are a smoke test. They are too short for a policy you would play back. The seed is 42.

```bash
uv run python scripts/train.py --task rough --num-envs 2048 --max-iterations 2500
```

The state-machine policy steps at 10 Hz. The PD still retargets at 50 Hz inside that step. On an RTX 5070 Ti with 64 environments, the same simulated time costs about half as much wall time as a 50 Hz policy step. Check a checkpoint on rough-test every 500 updates:

```bash
uv run python scripts/eval_state_machine.py --controller checkpoint --mode table \
  --task rough-test --checkpoint logs/rsl_rl/walter_fs_state_machine_rough/<run>/model_500.pt
```

A scripted climber that tumbles only while the footprint gate is open is the feasibility check. It cleared the 0.22 m inverted stairs and scored above both always-drive and always-tumble:

```bash
uv run python scripts/eval_state_machine.py --controller oracle --mode probe --device cuda:0
```

The 16-D teacher stack:

```bash
uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500
```

Continue a checkpoint. `--max-iterations` counts additional updates. The new files go in a fresh timestamp directory under the log root.

```bash
uv run python scripts/train.py --task rough-test --resume logs/rsl_rl/.../model_500.pt --max-iterations 500
```

Scalars land in the run directory. View them with:

```bash
uv run tensorboard --logdir logs/rsl_rl
```

The panel serves play and train controls on localhost and runs one child process at a time:

```bash
uv run python scripts/panel.py
```

Open `http://127.0.0.1:8765`. `--port` changes the port.

Record one rough-test cell (PD teacher, or a 16-D checkpoint on `TeacherStudentEnv`):

```bash
uv run python scripts/eval_cells.py --controller pd --cell flat --difficulty medium
uv run python scripts/eval_cells.py --controller checkpoint --cell pyramid-stairs \
  --difficulty hard --checkpoint logs/rsl_rl/.../model_500.pt
```

In the Viser **Twist** folder during PD or smooth-tumble playback:

1. Check **Enable** so the sliders override the sampled velocity command.
2. Set **lin_vel_x** for forward speed and **ang_vel_z** for yaw rate. Lateral speed is locked at zero.
3. Check **Tank Mode** to apply **hip_angle** and **knee_angle**. It starts unchecked. Unchecked, both commands are zero and the sliders keep their values. **hip_angle** is degrees; positive points the front hips forward and the rear hips backward. **Flip front hip** reverses the front hip and front knee zeros. **knee_angle** is degrees and shifts the knee zero; positive lowers the front inner wheels. With Tank Mode on, the rear zero moves the other way, so positive raises the rear inner wheels. The knee servo uses `tank_actuator_kp` and `tank_actuator_kd` while the box is checked.
4. Leave **Tumbling** checked so the stall integral runs. The motors tab shows **lin vel integral (m)**. Unchecked, the integral is cleared and the knees stay on the roll pose. A state-machine checkpoint ignores this checkbox; the network latch replaces the integral.
5. Leave **Orientation** checked to level pitch and roll. Unchecked, those errors are treated as zero.
6. **Rated torque cap** and **Rated speed cap** hold the DC motors on the continuous ratings. Both start unchecked, so the peak torque envelope and the no-load speed corner stay in use.
7. Keyboard: **I** / **K** step forward speed, **J** / **L** step yaw rate. **Zero** clears the velocity sliders and the pose sliders.

Build a PD environment in Python when you do not want the viewer loop:

```python
from walter_rl.control.full_body_pd.env import make_pd_env
from walter_rl.control.full_body_pd.policy import FullBodyPdPolicy

env = make_pd_env("rough", play=True, num_envs=1)
policy = FullBodyPdPolicy(env)
action = policy(env.reset()[0])
env.close()
```

`policy` ignores the observation tensor. It reads the robot state and the `twist` command.

## Configuration

There is no `.env` file. Behavior comes from CLI flags, `gains.py`, and two environment variables read by mjlab.

### Playback CLI

| Flag | Default | Choices / meaning |
| --- | --- | --- |
| `--policy` | `pd` | `play.py` only. `pd` or `checkpoint`. |
| `--checkpoint` | none | Required with `--policy checkpoint`. A `model_*.pt` file from an rsl-rl run. |
| `--task` | `flat` | `flat`, `rough`, `rough-test`, `stairs` |
| `--num-envs` | `1` | Number of parallel environments |
| `--viewer` | `viser` | `viser` (browser) or `native` (MuJoCo passive viewer) |
| `--payload` | none | Rigid mass welded to the torso top. `rider` is a 75 kg seated human. Used by the PD and smooth-tumble scripts. |

| Task | Scene |
| --- | --- |
| `flat` | Infinite plane. The PD task drops terrain height scans. Training puts a scan back so the observation width matches the other scenes. The state machine then replaces that scan with the front and rear grids. |
| `rough` | Training grid for state machine and teacher. Stairs run from 0.06 m to 0.22 m over 10 rows. Slopes stay at 0.45 and waves at 0.12 m. A cube column uses the rough-test height cap, a 2 m center platform, and a 1 m apron. `random_rough` scales with the row. Robots spawn at the tile center and keep their column. A row promotes when the root reaches the exit radius, and demotes on a fall, a hard belly hit, or a long episode that never leaves the inner half of that radius. The top row re-rolls. |
| `rough-test` | The shared rough mix on every third difficulty row, with stair step heights scaled by 1.5 (hard row 0.18 m). The uniform-noise column is a random grid of cubes. It is the evaluation grid, not the state-machine training grid. Training on it still starts on row 0. A finished or failed climb returns to the column the environment was born on. |
| `stairs` | Flat (15%) plus pyramid and inverse-pyramid stairs (6–18 cm). The teacher and the 50 Hz tasks promote along the spawn yaw, and 80% of inverse-stair environments stay on the hardest row. |

Play mode (`play=True`, which the play scripts set) uses a very long episode, disables observation noise and pushes, and turns the terrain curriculum off for the run.

### Training CLI

| Flag | Default | Meaning |
| --- | --- | --- |
| `--task` | `flat` | Same four scenes as playback. |
| `--num-envs` | `8` | Parallel environments. The panel default is 1024. |
| `--max-iterations` | `2` | PPO updates. On `--resume`, this is the number of additional updates. |
| `--teacher` | off | Train the 17-D teacher-student policy. Off selects the tumble toggle and tank hip. |
| `--teacher-only-reward` | off | Zero every reward except PD action matching. Requires `--teacher`. With the match weight at 0 this leaves the run with no reward. |
| `--lin-vel-x-min` / `--lin-vel-x-max` | `0.25` / `1.25` | Forward command bounds, m/s. Cube-root sampling still averages 1 m/s inside the default span. |
| `--ang-vel-z-min` / `--ang-vel-z-max` | `-1` / `1` | Yaw command bounds, rad/s. 75% of resamples are still zero, and 5% are a full stop. |
| `--device` | GPU if present | `cuda:0` or `cpu`. |
| `--log-root` | `logs/rsl_rl` | Directory for new runs. Checkpoints load through `--resume` with an explicit path. |
| `--resume` | none | Checkpoint file. Weights, optimizer, and the iteration counter are restored. |

Log folder names are `walter_fs_state_machine` and `walter_fs_teacher`. Any task other than `flat` appends `_<task>`, for example `walter_fs_state_machine_rough-test`. Checkpoints are `model_<iteration>.pt`, saved every 50 updates.

PPO settings live in `src/walter_rl/rl/ppo.py`. The teacher runner steps at 50 Hz, uses 64 steps per environment, entropy coefficient `0.005`, a Gaussian actor, and the same MLP widths as the state machine (actor 256/128/64, critic 512/256/128). It shares the front and rear height grids with the state machine. The state-machine runner steps at 10 Hz, uses 24 steps (2.4 s), entropy `0.003`, and a Beta actor on `[-1, 1]`. Both use 5 epochs, 4 minibatches, adaptive Adam at `1e-3`, discount `0.99`, GAE `0.95`, and clip `0.2`. The logger is TensorBoard. Checkpoints from the old torso-scan teacher runs do not load.

### State machine

The policy action is read once per 0.1 s step. `policy_action[:, 0] > 0` requests tumble. A rising edge holds tumble on for 0.5 s (5 policy steps). A falling edge holds the stored output negative for 0.5 s so a new tumble cannot start. `policy_action[:, 1]` is on `[-1, 1]` and maps to a Tank Mode hip offset from 0° to 30° by `(a + 1) / 2`. Positive points the front hips forward and the rear hips back. Pitch leveling still adds on top of that offset. Tank Mode stays on, and the tank knee command is rewritten to 0 every step. The PD controller then fills hips, knees, wheels, and a zero waist command, five times per policy step, integrating the knee reference at 0.02 s. Episode reset clears the latch, the dwell counters, the tumble spin reference, and the hip offset.

The actor sees:

- `height_scan_front` on the head, yaw-aligned, near edge 0.1 m past the head box
- `height_scan_rear` on the torso, yaw-aligned, near edge 0.1 m past the torso box
- `projected_gravity` (noise ±0.05)
- `base_ang_vel`, the IMU gyro (noise ±0.2)
- `hip_pos`, biased hip encoders
- `knee_sincos`
- `joint_torques`, hip and knee torque divided by each joint's continuous rating
- `wheel_torques`, axle torque divided by each wheel's continuous rating
- `command`, the twist
- `last_policy_action`, the latched toggle and the mapped hip
- `tumble_applied`, the applied bit over the last 0.5 s, oldest first

The critic sees the same terms without noise, plus body linear velocity, knee and wheel rates, the footprint height profile, a terrain-column one-hot with the normalized row, and the remaining on- and off-dwell. Actor scans add uniform noise of ±0.1 before the height is scaled by the 5 m ray length. `StateMachineVecEnv` tells rsl-rl the action dimension is 2. The environment still steps a 16-D PD action.

Reward weights are the constants at the top of `src/walter_rl/tasks/velocity/config/walter_fs/env_cfgs.py`. The `SM_*` constants are the 10 Hz state-machine table. Zero-weight terms stay in the table and are skipped at runtime. Live terms are forward tracking, forward progress, yaw tracking, uprightness, action rate, climb tumble, unneeded tumble, downhill tumble, a light belly-contact penalty, limb torque, and stuck forward speed. Climb tumble pays for forward knee rate while a riser is within 0.35 m ahead of the front axles or still under the wheelbase. The gate is shut at a 4 cm rise and fully open at 8 cm, so a 0.45 slope does not open it. Unneeded tumble charges the latch when no such rise sits between the rear axles and 0.75 m past the front axles. Downhill tumble charges nose-down pitch, in radians, while the latch is on and that same span falls away. A hard belly hit, about 300 N, ends the episode. Lighter contact pays the penalty instead.

### Teacher stack

`TeacherStudentEnv` snapshots `FullBodyPdController.act()` before physics. On a closed forward-scan match gate the stored target is that reactive action. On an open gate the stored target is the same command with tumble forced on. Training steps the student action. `follow_teacher` steps the stored target instead, which is how `eval_cells.py` plays the PD teacher.

`track_teacher` is an exponential match of that stored action at weight `0.5` (kernel in `[0, 1]`). `--teacher-only-reward` zeros every other term and leaves this one active.

### Controller gains

Edit `src/walter_rl/control/full_body_pd/gains.py` for the PD inner loop, and `src/walter_rl/control/smooth_tumble/gains.py` for the smooth-tumble player. Each nested dataclass is one loop. Angles are radians, rates are rad/s, and linear speeds are m/s. Hip, knee, and waist actuator gains are applied when the environment is built, so restart the process after changing them. Tank Mode knee gains (`tank_actuator_kp`, `tank_actuator_kd`) switch as soon as the checkbox changes.

| Group | Role | Notable defaults |
| --- | --- | --- |
| `PitchGains` | Level torso pitch with hip splay | `kp=6`, `kd=0.4`, actuator `800 / 100` |
| `RollGains` | Drop the downhill inner wheels | `kp=8`, `kd=0.5`, `q_max=π/2`, actuator `250 / 50`, tank actuator `250 / 50` |
| `VelocityGains` | Track forward speed on all eight wheels | `kp=15`, `full_speed_command=10` m/s, `wheel_radius=0.10`, extra speed clip `50` rad/s |
| `YawRateGains` | Skid-steer from `ang_vel_z` | `kp=5`, half-width `0.32` m, yaw limit `2.5` rad/s |
| `TumbleGains` | Integrate missed forward distance, then spin the knees | `threshold=0.2` m, `release=0.1` m, `cap=0.5` m, `leak=2.5` /s, `min_forward_cmd=0.2` m/s, `min_speed_error=0.6` m/s, `rate=3` rad/s, `wheel_speed=0` rad/s |
| `WaistGains` | Hold `head_joint` at zero | `kp=2000`, `kd=50`, effort `288` N·m |

`sign` is `+1` or `-1`. Flip it if that loop runs away.

`make_pd_env` also accepts `device`, `gains`, `easy_rough` (one 2–4 cm rough tile), `lock_speed`, `lock_heading`, `episode_length_s`, and `payload` (a name from `walter_rl.assets.payloads`, or none).

### Integral tumble

`TumbleGains` is the stall loop used by PD playback and by smooth tumble. The state-machine latch replaces it during training and checkpoint playback. Its state is missed forward distance, in meters. The Viser motors tab draws that state as **lin vel integral (m)**, with the bar scaled to `cap`.

Each control step reads body-frame forward speed, `v_x`:

- When `lin_vel_x` is above `min_forward_cmd` and `lin_vel_x - v_x` is above `min_speed_error`, the state adds that speed error times the step `dt`. The integral grows only from a forward shortfall.
- Otherwise the state leaks toward zero as `lag * exp(-leak * dt)`. `leak` is in 1/s.
- The state is clamped at `cap`, so a long stall cannot leave a large tail. Keep `cap` above `threshold`.

A hysteresis gate reads that state after the update. Tumbling turns on above `threshold` and stays on until the integral falls below `release`. While it is on, every knee reference is captured at the measured angle and then advances at `rate` about +Y. Positive +Y drops the front wheel and lifts the rear. The knee position servo remains the inner loop. Tumble replaces the roll angle and, while the gate is on, the wheel commands.

Every wheel is written to `tumble.wheel_speed` for that interval. The default is 0 rad/s. Positive matches a positive forward command. Speed tracking and skid steering resume as soon as the gate turns off, including while the knees settle.

After release, each knee continues forward at `rate` until it reaches the roll pose on the revolution latched from the spin reference. The latch is taken when tumbling ends, so a shin that passes the pose stays on that turn. Unchecking **Tumbling** clears the integral, the spin reference, and the settle. An episode reset does the same.

### Environment variables

| Variable | Default | Effect |
| --- | --- | --- |
| `MUJOCO_GL` | `egl` on Linux | Rendering backend. mjlab sets this before MuJoCo imports. Use `osmesa` on machines without EGL, or `glfw` for a windowed GL context. |
| `MJLAB_WARP_QUIET` | unset (`0`) | Set to `1`, `true`, or `yes` to hide Warp kernel compile logs. |

# File Explanations (Directory Structure)

```text
gen0WaLTER/
├── pyproject.toml
├── uv.lock
├── README.md
├── SIMULATION.md
├── RL Manual.md
├── scripts/
│   ├── train.py
│   ├── play.py
│   ├── play_pd.py
│   ├── play_smooth_tumble.py
│   ├── panel.py
│   ├── eval_cells.py
│   ├── eval_state_machine.py
│   └── check_teacher_reward.py
├── renders/              # MJCF preview images
├── src/walter_rl/
│   ├── assets/
│   │   ├── motor_models/
│   │   ├── payloads/
│   │   └── walter_fs/
│   ├── control/
│   │   ├── full_body_pd/
│   │   └── smooth_tumble/
│   ├── panel/
│   ├── rl/
│   │   ├── state_machine_env.py
│   │   ├── teacher_env.py
│   │   ├── vecenv.py
│   │   ├── ppo.py
│   │   └── checkpoint_policy.py
│   ├── tasks/velocity/
│   │   ├── mdp/
│   │   └── config/walter_fs/env_cfgs.py
│   └── viewer/
├── logs/rsl_rl/          # PPO checkpoints, TensorBoard, eval cell clips (gitignored)
└── mjlab/
```

`.venv/` is the local uv environment and is omitted above. `logs/rsl_rl/` holds training runs and cell clips.

## Component Breakdown

- **`pyproject.toml`** — Package `walter_rl` 0.1.0. Depends on the local `mjlab` checkout, `rsl-rl-lib` 5.4.2, TensorBoard, and `tensordict`.
- **`uv.lock`** — Locked dependency graph, including PyTorch, MuJoCo, Warp, Viser, and rsl-rl.
- **`scripts/train.py`** — PPO entry point. Default env is `StateMachineEnv`. `--teacher` selects `TeacherStudentEnv`.
- **`scripts/play.py`** — PD playback, or a state-machine or teacher checkpoint in Viser or the native viewer.
- **`scripts/play_pd.py`** — PD playback only.
- **`scripts/play_smooth_tumble.py`** — Smooth-tumble playback. Hips cancel the knee's ride-height change while tumbling.
- **`scripts/panel.py`** — Browser panel. Play choices are classical PD, smooth tumble, and a checkpoint. Train choices include the teacher flags and a resume path under `logs/rsl_rl`.
- **`scripts/eval_cells.py`** — One rough-test cell, 8 s, forward command 1 m/s. Writes a clip and `frames.csv` for the PD teacher or a teacher checkpoint.
- **`scripts/eval_state_machine.py`** — Rough or rough-test cells for the 2-D state-machine policy (oracle, checkpoint, or probe/table modes).
- **`scripts/check_teacher_reward.py`** — Short rollout that compares the teacher action with zeros on the match term.
- **`SIMULATION.md`** — Path from the MJCF through mjlab and MuJoCo Warp to a DC-motor torque and the PD loops.
- **`src/walter_rl/rl/state_machine_env.py`** — Latches the tumble toggle, holds each edge for 0.5 s, writes the 0–30° tank hip, and steps the PD action.
- **`src/walter_rl/rl/teacher_env.py`** — Stores the scan-conditioned PD target, then steps the student or that target.
- **`src/walter_rl/rl/vecenv.py`** — rsl-rl 5.4 adapter. `StateMachineVecEnv` reports two actions.
- **`src/walter_rl/rl/ppo.py`** — `OnPolicyRunner` configs: Beta actor for the state machine, Gaussian for the teacher, shared critic widths.
- **`src/walter_rl/rl/checkpoint_policy.py`** — Viewer adapter. Reads the `actor` observation group and reports the stored tumble decision and tank hip offset.
- **`src/walter_rl/control/full_body_pd/`** — Pitch, roll, wheel speed, yaw, integral tumble, and the external tumble latch.
- **`src/walter_rl/control/smooth_tumble/`** — Same loops, with level hips during tumble and settle.
- **`src/walter_rl/tasks/velocity/config/walter_fs/env_cfgs.py`** — Scene, command, reward, and observation builders, including `walter_fs_state_machine_env_cfg` and `walter_fs_teacher_env_cfg`.
- **`src/walter_rl/tasks/velocity/mdp/`** — Rewards, finite-safe observations, terrain curricula, and the rough-test drive/climb curriculum.
- **`src/walter_rl/panel/`** — Local HTTP server, the page in `index.html`, allowlisted commands, and a single job supervisor.
- **`src/walter_rl/assets/`** — MJCF, CubeMars motor models, and the `rider` payload. Motor curves are documented in [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md).
- **`mjlab/`** — Vendored simulation library (version 1.6.0 in this tree): manager-based RL env, MuJoCo Warp sim, terrain generator, velocity command, and the Viser and native viewers.

# Developer Notes

## System Architecture

The code splits into four layers.

**Robot description.** `get_spec()` loads `walter_fs.xml` and prepares it for mjlab. `get_walter_fs_robot_cfg()` returns an `EntityCfg` whose articulation is three DC motor groups from `motor_models`: position servos on `.*_thigh_joint` and `.*_thigh_shin_joint`, and a velocity servo on `.*wheel_joint`. Ratings, the torque-speed curve, and the extra joint gears are documented in [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md). Training-time hip and knee stiffness is 20 N·m/rad. The PD environment replaces those with the stiffer gains in `gains.py` and adds a position actuator on `head_joint`. The PD knee action is an absolute angle.

**Scenes.** `env_cfgs.py` starts from mjlab's `make_velocity_env_cfg()` and retargets it to WaLTER. The physics action vector is hip position, knee position, and wheel velocity. A torso ray cast, a small body-height scan, a belly contact sensor, and one contact sensor per wheel pair feed rewards and terminations. `walter_fs_rl_env_cfg` is the shared training scene: simple twist sampling, the tracking rewards, a waist lock, and a terrain scan. The state-machine config replaces the scan and the reward table. The teacher config adds the match term and the predictive tumble term on the torso scan. There is no `mjlab.tasks` entry point; `scripts/train.py` builds the env directly.

**Inner controller.** `FullBodyPdController.act()` fills one action row per environment:

1. Pitch PD on `projected_gravity_b.x` swings the low-end hips outward from vertical, then adds the joystick hip angle.
2. Roll PD on `projected_gravity_b.y` folds the downhill knees until the inner wheel is straight down. Tank Mode flips the rear knee zero.
3. The stall integral, when it is the active gate, spins the knees at `tumble.rate` and holds the wheels at `tumble.wheel_speed`. `set_tumble_command` replaces that gate with an external boolean mask and still runs knee spin and settle.
4. Wheels scale `lin_vel_x` so `full_speed_command` requests the motor's maximum wheel rate, add proportional speed error, and add a differential from the yaw-rate loop. The tumble gate replaces that command with `tumble.wheel_speed`.
5. The waist column is written to zero.

**Learned policy.** `StateMachineEnv` is the default (2-D policy, 10 Hz). `TeacherStudentEnv` is the 17-D alternative (50 Hz). `RslRlVecEnvWrapper` and the specialized vec wrappers present the mjlab env to rsl-rl 5.4, including timeout bootstrapping. `CheckpointPolicy` matches the viewer protocol (`reset` and `__call__`). Simulation steps run in PyTorch on the device chosen by the script. The native viewer copies one environment's state back to CPU `MjData`. Viser renders through `mjviser`.

## Development Setup

Install the editable checkout and run a short train or a play script:

```bash
uv sync
uv run python scripts/train.py --num-envs 8 --max-iterations 2
uv run python scripts/play_pd.py --task flat --num-envs 1
```

`uv sync` installs both packages in editable mode (`walter_rl` from `src/`, `mjlab` from `mjlab/`). Changes under `src/walter_rl` and `mjlab/src/mjlab` apply on the next process start. Warp compiles kernels on first launch; set `MJLAB_WARP_QUIET=1` to suppress that log.

There is no database and no separate compile step. For a headless PD check that does not open a viewer, construct `make_pd_env(..., play=False)` and call `policy` once, then `env.close()`.

## Testing

This repository has no `tests/` directory and no pytest dependency. Two scripts check pieces of the teacher path by running them:

- `uv run python scripts/check_teacher_reward.py` checks that the match term prefers the teacher action over zeros and that `TRACK_TEACHER_WEIGHT` stays below forward plus yaw tracking.
- `uv run python scripts/eval_cells.py` writes a clip for one rough-test cell. State-machine and teacher checkpoints are not interchangeable between `play.py` and `eval_cells.py` without matching the env each script builds.
- `uv run python scripts/eval_state_machine.py --controller oracle --mode probe` rolls the scripted climber on the top inverted-stair row and compares its return with always-drive and always-tumble. `--mode table` scores every cell of `--task`.

## Contributing & Future Work

There is no `CONTRIBUTING` guide. A practical change is a focused edit plus a play session on `flat` and one uneven task (`rough` or `stairs`):

1. Fork or branch from `[INSERT REPOSITORY URL HERE]`.
2. Keep controller numbers in `gains.py`. Keep scene layout and reward weights in `env_cfgs.py`.
3. Run `uv run python scripts/play.py --policy pd --task flat` and one of `--task rough` or `--task stairs`.
4. Open a pull request against `[INSERT DEFAULT BRANCH HERE]`.

Limits visible from the current tree:

- Teacher matching and wheel-pair collision weights live with the other `FORWARD_*` constants in `env_cfgs.py`. The state-machine reward table still names `sensor_name="terrain_scan"` for `track_teacher`; the match function ignores the argument.
- `scripts/play.py --policy checkpoint` loads state-machine or teacher weights from the checkpoint path (`walter_fs_teacher` selects the 17-D stack).
- `src/walter_rl/tasks/__init__.py` and `src/walter_rl/control/__init__.py` still describe a PD-only package. Training imports the PD controller from `walter_rl.rl`.
- The root `pyproject.toml` description still says the package is full-body PD playback.
- `RL Manual.md` is a second manual and still describes the 16-D student as the default, with a teacher match weight of 0.5. This README is the one that matches the code.
- State-machine checkpoints live under `logs/rsl_rl/walter_fs_state_machine_*`.
- `mjlab/src/mjlab/utils/lab_api/math.py` and `mjlab/src/mjlab/utils/spec.py` each contain an upstream note to double-check a quaternion product and a spec helper. Those are inside the vendored library.
- License, CI, and an automated test suite are not in the tree. Build-status and license badges above are placeholders until those exist.

## License

No `LICENSE` file is present. The project license is **[INSERT LICENSE HERE]**.
