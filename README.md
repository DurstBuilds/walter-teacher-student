# gen0WaLTER

![License](https://img.shields.io/badge/License-INSERT__LICENSE-blue)
![Build Status](https://img.shields.io/badge/Build-INSERT__STATUS-lightgrey)
![Version](https://img.shields.io/badge/Version-0.1.0-blue)

**gen0WaLTER** is a simulation package for the full-scale WaLTER robot. It drives the robot with a classical full-body PD controller inside MuJoCo scenes for flat ground, rough terrain, and stairs. The package version is **0.1.0**.

WaLTER is a wheeled leg platform: four hip joints, four continuous knee joints, eight driven wheels, and a waist (`head_joint`) held at zero. Balancing, speed tracking, and skid steering are closed in code. A forward-speed integral counts meters the body falls behind `lin_vel_x`. Once that distance crosses a threshold, the knees spin about +Y, the wheels hold `tumble.wheel_speed`, and the legs climb the stall.

The primary use case is interactive playback. `scripts/play_pd.py` builds one of the four scenes and steps `FullBodyPdPolicy` in the Viser browser viewer or MuJoCo's native viewer. Velocity-task observations, rewards, and a terrain curriculum live beside the controller so the same robot description can support later reinforcement-learning work. That control package is not registered as an RL task.

How an MJCF file becomes a joint torque is in [`SIMULATION.md`](SIMULATION.md). Masses and link lengths are in [`src/walter_rl/assets/walter_fs/README.md`](src/walter_rl/assets/walter_fs/README.md).

## Key Features

- Full-body PD on hips, knees, wheels, and the waist, with gains collected in one dataclass file.
- Pitch leveling by splaying the downhill hips, and roll leveling by extending the inner legs.
- Wheel velocity tracking: `full_speed_command` requests the motor's maximum wheel rate, with proportional speed error on top, plus a skid-steer yaw loop.
- Integral tumble: missed forward distance spins every knee about +Y between `threshold` and `release`, and holds every wheel at `tumble.wheel_speed`. The knees then settle forward onto the roll pose while speed tracking resumes. The Viser motors tab draws the state as **lin vel integral (m)**.
- Four scenes: `flat`, `rough`, `rough-test`, and `stairs`, including a shortened rough grid with taller stairs.
- Checked-in full-scale MJCF (`walter_fs.xml`), with generator motors stripped and CubeMars DC motor actuators attached.
- Actuator datasheets and torque-speed curves in `motor_models`. Details are in [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md).
- Viser joystick for forward speed, yaw rate, hip angle, knee angle, Tank Mode, and the tumble integral.
- Batched simulation through mjlab, MuJoCo Warp, and PyTorch, on CUDA when a GPU is present and on CPU otherwise.

# User Manual

## Prerequisites

| Requirement | Version / notes |
| --- | --- |
| Python | `>=3.12,<3.14` (the checked-in environment uses **3.13.13**) |
| [uv](https://docs.astral.sh/uv/) | **0.11.19** or newer (`uv_build>=0.12.1,<0.13`) |
| Operating system | Linux is the supported path. mjlab sets `MUJOCO_GL=egl` on Linux before MuJoCo imports. |
| GPU | Optional NVIDIA GPU. `make_pd_env` selects `cuda:0` when `torch.cuda.is_available()`, otherwise `cpu`. |
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

## Installation

Clone the repository and sync the lockfile. `walter_rl` depends on the vendored `mjlab` tree as an editable path source.

```bash
git clone [INSERT REPOSITORY URL HERE]
cd gen0WaLTER
uv python install 3.13
uv sync
```

`uv sync` creates `.venv` and installs `walter_rl` and `mjlab` in editable mode. CUDA and CPU PyTorch indexes are declared as optional extras on `mjlab` (`cu128` and `cpu`). The root project does not select either extra; the lockfile resolves the default `torch==2.14.0` wheel.

## Quick Start / Usage

Play the PD controller on flat ground. Viser prints a local URL when the server starts; open that URL in a browser.

```bash
uv run python scripts/play_pd.py --task flat
```

Other scenes:

```bash
uv run python scripts/play_pd.py --task rough --viewer viser
uv run python scripts/play_pd.py --task rough-test --viewer viser
uv run python scripts/play_pd.py --task stairs --viewer native
uv run python scripts/play_pd.py --task flat --num-envs 4
uv run python scripts/play_pd.py --task flat --payload rider
```

In the Viser **Twist** folder:

1. Check **Enable** so the sliders override the sampled velocity command.
2. Set **lin_vel_x** for forward speed and **ang_vel_z** for yaw rate. Lateral speed is locked at zero.
3. Check **Tank Mode** to apply **hip_angle** and **knee_angle**. It starts unchecked. Unchecked, both commands are zero and the sliders keep their values. **hip_angle** is degrees; positive points the front hips forward and the rear hips backward. **Flip front hip** reverses the front hip and front knee zeros (like Tank Mode for the rear knee). **knee_angle** is degrees and shifts the knee zero; positive lowers the front inner wheels. With Tank Mode on, the rear zero moves the other way, so positive raises the rear inner wheels. The knee servo uses `tank_actuator_kp` and `tank_actuator_kd` while the box is checked.
4. Leave **Tumbling** checked so the stall integral runs. The motors tab shows **lin vel integral (m)**. Unchecked, the integral is cleared and the knees stay on the roll pose.
5. Leave **Orientation** checked to level pitch and roll. Unchecked, those errors are treated as zero.
6. **Rated torque cap** and **Rated speed cap** hold the DC motors on the continuous ratings. Both start unchecked, so the peak torque envelope and the no-load speed corner stay in use.
7. Keyboard: **I** / **K** step forward speed, **J** / **L** step yaw rate. **Zero** clears the velocity sliders and the pose sliders.

Build an environment in Python when you do not want the viewer loop:

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

### CLI

| Flag | Default | Choices / meaning |
| --- | --- | --- |
| `--task` | `flat` | `flat`, `rough`, `rough-test`, `stairs` |
| `--num-envs` | `1` | Number of parallel environments |
| `--viewer` | `viser` | `viser` (browser) or `native` (MuJoCo passive viewer) |
| `--payload` | none | Rigid mass welded to the torso top. `rider` is a 75 kg seated human. |

| Task | Scene |
| --- | --- |
| `flat` | Infinite plane. Terrain height scans are removed. |
| `rough` | mjlab velocity mix with WaLTER caps: pyramid slopes up to 0.45, waves up to 0.12 m, stairs up to 0.12 m. |
| `rough-test` | Same mix on every third difficulty row (0, 3, 6, 9), with stair step heights scaled by 1.5. The uniform-noise column is a random grid of cubes whose cell height grows from 0 to 0.24 m along the row (four fifths of the 0.3 m preset). |
| `stairs` | Flat (15%) plus pyramid and inverse-pyramid stairs (6–18 cm). The curriculum holds an environment on the hardest row after it clears it. |

Play mode (`play=True`, which the script always sets) uses a very long episode, disables observation noise and pushes, and turns the terrain curriculum off for the run.

### Controller gains

Edit `src/walter_rl/control/full_body_pd/gains.py`. Each nested dataclass is one loop. Angles are radians, rates are rad/s, and linear speeds are m/s. Hip, knee, and waist actuator gains are applied when the environment is built, so restart the process after changing them. Tank Mode knee gains (`tank_actuator_kp`, `tank_actuator_kd`) switch as soon as the checkbox changes.

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

`TumbleGains` is the stall loop. Its state is missed forward distance, in meters. The Viser motors tab draws that state as **lin vel integral (m)**, with the bar scaled to `cap`.

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
├── scripts/
│   └── play_pd.py
├── src/walter_rl/
│   ├── __init__.py
│   ├── assets/
│   │   ├── motor_models/
│   │   │   └── README.md
│   │   ├── payloads/
│   │   │   ├── __init__.py
│   │   │   └── rider.xml
│   │   └── walter_fs/
│   │       ├── README.md
│   │       ├── walter_constants.py
│   │       └── walter_fs.xml
│   ├── control/full_body_pd/
│   │   ├── controller.py
│   │   ├── env.py
│   │   ├── gains.py
│   │   └── policy.py
│   └── tasks/velocity/
│       ├── mdp/
│       │   ├── curriculums.py
│       │   ├── observations.py
│       │   └── rewards.py
│       └── config/walter_fs/
│           └── env_cfgs.py
└── mjlab/
    ├── pyproject.toml
    └── src/mjlab/
        ├── actuator/
        ├── envs/
        ├── managers/
        ├── sensor/
        ├── sim/
        ├── tasks/velocity/
        ├── terrains/
        └── viewer/
```

`.venv/` is the local uv environment and is omitted above.

## Component Breakdown

- **`pyproject.toml`** — Package `walter_rl` 0.1.0. The only dependency is the local `mjlab` checkout.
- **`uv.lock`** — Locked dependency graph, including PyTorch, MuJoCo, Warp, and Viser.
- **`scripts/play_pd.py`** — CLI entry point. Builds a play environment, wraps it in `FullBodyPdPolicy`, and runs Viser or the native viewer.
- **`SIMULATION.md`** — Path from the MJCF through mjlab and MuJoCo Warp to a DC-motor torque and the PD loops.
- **`src/walter_rl/assets/walter_fs/walter_fs.xml`** — Checked-in full-scale MJCF produced by `generate-walter --fs`.
- **`src/walter_rl/assets/walter_fs/README.md`** — Mass table, link lengths, and which CubeMars motor drives each joint.
- **`src/walter_rl/assets/motor_models/`** — CubeMars actuator datasheets and output-side torque-speed curves. See [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md) for the catalog, curve, and how to add a motor.
- **`src/walter_rl/assets/payloads/`** — Rigid masses welded to the torso for play. `rider` is the 75 kg seated human selected by `--payload rider`.
- **`src/walter_rl/assets/walter_fs/walter_constants.py`** — Loads that MJCF, limits hips to ±70°, replaces wheel contact parameters, adds tire and spoke visuals, adds belly and leg contact bumps plus a torso IMU, strips generator motors, and attaches the hip, knee, and wheel motors from `motor_models`.
- **`src/walter_rl/control/full_body_pd/gains.py`** — Tunable PD numbers. This is the file to edit when a loop oscillates or runs away.
- **`src/walter_rl/control/full_body_pd/controller.py`** — `FullBodyPdController` writes the action vector: pitch hips, roll knees, integral tumble override, wheel speeds, and a zero waist command. Tank Mode flips the rear knee zero. While tumbling, every wheel holds `tumble.wheel_speed`.
- **`src/walter_rl/control/full_body_pd/policy.py`** — Viewer adapter. `__call__` drops the observation and returns `controller.act()`. `reset` clears the tumble integral.
- **`src/walter_rl/control/full_body_pd/env.py`** — `make_pd_env` clones a velocity scene, stiffens the hip and knee servos, adds the waist position actuator, and can lock the twist command.
- **`src/walter_rl/tasks/velocity/config/walter_fs/env_cfgs.py`** — Flat, rough, rough-test, and stairs `ManagerBasedRlEnvCfg` builders: actions, commands, sensors, rewards, terminations, and play mode.
- **`src/walter_rl/tasks/velocity/mdp/rewards.py`** — WaLTER rewards: XY velocity tracking, climb `vz`, forward drive, ride height, head level, wheel-wheel contact, and terrain-gated knee tumble (forward spin rewarded, reverse spin and flat-ground waste penalized).
- **`src/walter_rl/tasks/velocity/mdp/observations.py`** — Finite-safe IMU, joint, and height-scan terms, plus sine/cosine knee angles for joints that can wind through full turns.
- **`src/walter_rl/tasks/velocity/mdp/curriculums.py`** — `terrain_levels_hold_max` promotes by progress along the commanded direction and stays on the hardest row once it is cleared.
- **`mjlab/`** — Vendored simulation library (version 1.6.0 in this tree): manager-based RL env, MuJoCo Warp sim, terrain generator, velocity command with the Viser joystick, and the Viser and native viewers.

# Developer Notes

## System Architecture

The code splits into three layers.

**Robot description.** `get_spec()` loads `walter_fs.xml` and prepares it for mjlab. `get_walter_fs_robot_cfg()` returns an `EntityCfg` whose articulation is three DC motor groups from `motor_models`: position servos on `.*_thigh_joint` and `.*_thigh_shin_joint`, and a velocity servo on `.*wheel_joint`. Ratings, the torque-speed curve, and the extra joint gears are documented in [`src/walter_rl/assets/motor_models/README.md`](src/walter_rl/assets/motor_models/README.md). Training-time hip and knee stiffness is 20 N·m/rad. The PD environment replaces those with the stiffer gains in `gains.py` and adds a position actuator on `head_joint` that the RL scene does not include. The PD knee action is an absolute angle. The velocity-task knee action is a relative step of `0.03` rad, which cannot hold a loaded leg assembly.

**Scenes.** `env_cfgs.py` starts from mjlab's `make_velocity_env_cfg()` and retargets it to WaLTER. The action vector is hip position, knee position, and wheel velocity. A torso ray cast, a small body-height scan, a belly contact sensor, and one contact sensor per wheel pair feed the reward and termination terms. `walter_rl.control` is intentionally not imported by a task registry. `make_pd_env` is the composition root for the controller.

**Controller.** `FullBodyPdController.act()` fills one action row per environment:

1. Pitch PD on `projected_gravity_b.x` swings the low-end hips outward from vertical, then adds the joystick hip angle.
2. Roll PD on `projected_gravity_b.y` folds the downhill knees until the inner wheel is straight down. Tank Mode flips the rear knee zero, so a positive **knee_angle** raises the rear inner wheels.
3. The stall integral adds `lin_vel_x - v_x` while the forward command and the speed error are both above their floors, and leaks toward zero otherwise. Above `tumble.threshold` every knee tracks a `+Y` spin at `tumble.rate`, and every wheel holds `tumble.wheel_speed`. Below `tumble.release` the knees settle forward onto the roll pose on the latched turn while the speed and yaw loops run again. See [Integral tumble](#integral-tumble).
4. Wheels scale `lin_vel_x` so `full_speed_command` requests the motor's maximum wheel rate, add proportional speed error, and add a differential from the yaw-rate loop. The tumble gate replaces that command with `tumble.wheel_speed`.
5. The waist column is written to zero.

`FullBodyPdPolicy` matches the viewer policy protocol (`reset` and `__call__`). Simulation steps run in PyTorch on the device chosen by `make_pd_env`. The native viewer copies one environment's state back to CPU `MjData` for MuJoCo's passive window. Viser renders through `mjviser`.

## Development Setup

Install the editable checkout and run the play script under the debugger or from a terminal:

```bash
uv sync
uv run python scripts/play_pd.py --task flat --num-envs 1
```

`uv sync` installs both packages in editable mode (`walter_rl` from `src/`, `mjlab` from `mjlab/`). Changes under `src/walter_rl` and `mjlab/src/mjlab` apply on the next process start. Warp compiles kernels on first launch; set `MJLAB_WARP_QUIET=1` to suppress that log.

There is no database and no separate compile step. Attach a Python debugger to `scripts/play_pd.py`. For a headless check that does not open a viewer, construct `make_pd_env(..., play=False)` and call `policy` once, then `env.close()`.

## Testing

This repository has no test suite, no `tests/` directory, and no test dependency in `pyproject.toml`. Nothing in the project is wired to pytest, unittest discovery, or a CI workflow.

## Contributing & Future Work

There is no `CONTRIBUTING` guide and no remote is recorded in this workspace. A practical change is a focused edit plus a play session on `flat` and one uneven task (`rough` or `stairs`):

1. Fork or branch from `[INSERT REPOSITORY URL HERE]`.
2. Keep controller numbers in `gains.py`. Keep scene layout in `env_cfgs.py`.
3. Run `uv run python scripts/play_pd.py --task flat` and one of `--task rough` or `--task stairs`.
4. Open a pull request against `[INSERT DEFAULT BRANCH HERE]`.

No `TODO` or `FIXME` comments are in `src/walter_rl` or `scripts/`. The open product gaps visible from the code are:

- The velocity MDP (actor/critic observations, rewards, terrain curriculum, command curriculum stages at step 0 and step `5000 * 24`) is configured, but no training script or `mjlab.tasks` entry point registers it.
- `src/walter_rl/control/__init__.py` states that the RL task registry does not import the PD package. A learned policy would be a separate entry point.
- `mjlab/src/mjlab/utils/lab_api/math.py` and `mjlab/src/mjlab/utils/spec.py` each contain an upstream note to double-check a quaternion product and a spec helper. Those are inside the vendored library, not WaLTER task code.
- License, CI, and an automated test suite are not in the tree. Build-status and license badges above are placeholders until those exist.

## License

No `LICENSE` file is present. The project license is **[INSERT LICENSE HERE]**.
