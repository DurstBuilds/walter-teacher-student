# Cycle 2 review

**Date:** 2026-10-14
**Checkpoint:** `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_00-33-54/model_998.pt` (checkpoint field `iter` is 998). Resume loaded `model_499.pt` at iteration 499 and ran 500 additional updates, iterations 499 through 998.
**Seed:** 42. `scripts/train.py` still sets `cfg.seed = 42`. No second seed.
**What changed:** the scan gate was not changed. Residual actions were not added. Scan size, scan offset, and the terrain mix were not changed.

**Train command, exit 0:**

`uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500 --resume research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-32-02/model_499.pt`

Logs: `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_00-33-54/`. 2048 environments, CUDA. No smoke training run, because the reward code was not edited.

**Clips:** `uv run python scripts/eval_cells.py --controller checkpoint --checkpoint research/logs/training/walter_fs_teacher_rough-test/2026-10-14_00-33-54/model_998.pt` with `--output` set per cell. Default 8 s, default 1 m/s command, eval seed 0. PD baselines were not rerecorded. PD numbers below are quoted from `research/logs/analysis/BASELINE_REVIEW.md`. Cycle 1 numbers are quoted from `research/logs/analysis/CYCLE1_REVIEW.md`.

| Clip | Directory |
| --- | --- |
| flat medium | `research/logs/analysis/cycle2_checkpoint_flat_medium` |
| flat hard | `research/logs/analysis/cycle2_checkpoint_flat_hard` |
| pyramid stairs hard | `research/logs/analysis/cycle2_checkpoint_pyramid-stairs_hard` |
| inverted stairs hard | `research/logs/analysis/cycle2_checkpoint_inverted-stairs_hard` |
| box grid hard | `research/logs/analysis/cycle2_checkpoint_box-grid_hard` |
| pyramid stairs hard, scan ablated | `research/logs/analysis/cycle2_checkpoint_pyramid-stairs_hard_ablate-scan` |

Frames below were taken from each `clip.mp4`. The numbers are from `frames.csv`.

**Onset used in the checks.** Same rule as cycle 1. The PD reference is the baseline gate distance. Student knee-rate onset is the first frame after 0.5 s with `knee_rate` above 1 rad/s. The first half-second is a startup transient.

## Why the gate was not changed

The cycle-2 hypothesis was that `forward_scan_step` keeps only the largest upward jump, so a downward inverted-stair face never opens `predictive_knee_tumble`, and the student keeps copying the reactive PD. The code does only count upward jumps: `_max_occupied_rise` takes `relu` of the difference between consecutive occupied bins, and the match gate is 1 at or below 0.04 m and 0 at or above 0.08 m. That half of the hypothesis is true of the function. It is not what the inverted-stair cell contains.

`forward_scan_height` in the cycle-1 inverted clip is mean clearance (more air is a larger number). It falls from 0.253 m at t = 0 to 0.215 m at t = 0.50 s, distance 0.470 m, speed still 0.996 m/s, before the stall at 0.943 m. A terrain drop would raise clearance. The fall is less air, so the terrain ahead is higher. Flat cruise in that same run sits near 0.250 m, so the dip is not the flat settling transient (flat settles from 0.274 m to 0.250 m).

A one-environment probe of the hard cells, PD playback, spawn at the patch origin, prints the gate the reward actually uses. On hard inverted stairs the forward bins are a flat floor and then a +0.180 m riser. At t = 0 the step measure is 0.180 m, the match gate is 0, and the tumble gate is 1. That stays true through 0.80 s of driving, while speed is about 0.83 m/s. The spawn is the pit floor (`z` about -0.79 m). The stairs in front of the robot go up. On hard pyramid stairs the same probe gives step 0.000 m, match gate 1, and tumble gate 0. Those bins step downward (about -0.29 m, then -0.47 m). The robot there starts on the top platform (`z` about 1.37 m) and the face ahead is a drop, which this gate ignores.

The failure mode in the hypothesis was the other cell. The gate already opens on hard inverted stairs, from the first frame, and the cycle-1 policy still started its knee-rate rise at 0.989 m after speed had gone negative. The gate was left as it is. Training continued from the same checkpoint.

## Reward curve

Scalars are from this run's TensorBoard file. Iteration 499 in this log is the first resumed update after `init_at_random_ep_len` restarts the episodes, so its mean reward (2.28) and episode length (39.6) are incomplete episodes, not the cycle-1 checkpoint's 48.61. By iteration 549 the episode statistics are back in the cycle-1 band. Mean reward then stays there. It does not decide the checks. `predictive_knee_tumble` stays small next to linear-velocity tracking.

| Iteration | Mean reward | Episode length | `track_teacher` | `predictive_knee_tumble` | `knee_stuck_tumble` | `track_linear_velocity` | Teacher action MSE | Terrain level mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 549 | 49.14 | 916.2 | 0.464 | 0.070 | 0.061 | 1.377 | 7.24 | 2.32 |
| 749 | 52.18 | 934.6 | 0.527 | 0.094 | 0.041 | 1.542 | 2.62 | 2.14 |
| 899 | 54.07 | 940.5 | 0.551 | 0.100 | 0.042 | 1.526 | 2.54 | 2.15 |
| 998 | 53.26 | 938.8 | 0.548 | 0.089 | 0.059 | 1.513 | 6.08 | 2.15 |

Iteration 997 mean reward was 55.41, so the last point is inside the same band as the late run, not a collapse. Terrain level mean ends at 2.15 of a max of 3. Cycle 1 ended at reward 48.61 and terrain level 2.11.

## Flat

Directories: `cycle2_checkpoint_flat_medium`, `cycle2_checkpoint_flat_hard`.

Both difficulties are flat, and the traces agree. The gate is 0 on every frame. Uprightness minimum is 0.999. After the first second, knee rate stays near zero.

Medium, after the first second: forward speed stays between 1.027 and 1.150 m/s (mean 1.093). Knee rate mean is 0.044 rad/s. One frame, at t = 6.60 s, reaches 0.107 rad/s. Every other frame after 1 s is under 0.1 rad/s. At t = 8.00 s, distance is 8.681 m, forward speed is 1.065 m/s, knee rate is 0.032 rad/s, uprightness is 0.999, forward-scan height is 0.255 m, and the tumble gate is 0. Knee-rate onset after 0.5 s does not occur. The startup spike peaks at 1.914 rad/s at t = 0.06 s and is under 1 rad/s by t = 0.20 s. End knee commands are about -0.28, -0.39, 0.05, -0.11 rad. They do not wind. Mean action MSE against the PD on this clip is 0.115.

Hard matches it. After the first second, speed stays between 1.043 and 1.154 m/s (mean 1.094) and knee rate stays under 0.099 rad/s (mean 0.044). End: distance 8.682 m, speed 1.108 m/s, knee rate 0.038 rad/s, uprightness 0.999, scan 0.255 m, gate 0. Mean action MSE is 0.114.

Command error is about +0.093 m/s on medium and +0.094 m/s on hard (means 1.093 and 1.094 against the 1.0 m/s command). The gap versus the PD cruise (mean 0.697 m/s in `BASELINE_REVIEW.md`) is about +0.40 m/s. Cycle 1 was a mean of 1.132 m/s with knee-rate means 0.025 and 0.024 rad/s. This run is slightly closer to the command and the knees are still quiet. The extra speed against the PD is the PD's undershoot, not a flat failure.

Frames at 1 s, 3 s, and 7 s show the body level, wheels on the plane, and no folded legs. Nothing in the frames looks like spinning in place or bouncing.

## Pyramid stairs, hard

Directory: `cycle2_checkpoint_pyramid-stairs_hard`.

The tumble gate stays 0 for all 400 frames. The robot pitches over the steps and is back on the flat cruise by the end. Final distance 8.345 m is past the PD's 5.492 m and in the same place as cycle 1's 8.662 m. End state is the student's flat cruise: speed 1.085 m/s, knee rate 0.039 rad/s, uprightness 0.999, scan 0.254 m, gate 0. Uprightness minimum is 0.908 at t = 2.84 s. Mean action MSE against the PD is 0.150.

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.959 | 1.034 | 0.631 | 0.999 | 0.550 | 0 |
| 1.26 | 1.235 | 1.192 | 1.040 | 0.988 | 0.662 | 0 |
| 2.00 | 1.953 | 1.146 | 1.341 | 0.924 | 0.779 | 0 |
| 3.00 | 2.892 | 1.127 | 1.571 | 0.928 | 0.465 | 0 |
| 5.00 | 5.063 | 1.109 | 0.045 | 0.999 | 0.255 | 0 |
| 8.00 | 8.345 | 1.085 | 0.039 | 0.999 | 0.254 | 0 |

Knee rate next exceeds 1 rad/s after the startup window at t = 1.26 s, distance 1.235 m, speed 1.192 m/s, scan 0.662 m, gate 0. It crosses 1 rad/s in short bursts through t = 3.52 s and is back under 0.05 rad/s by t = 4.50 s. Peak absolute knee command after 0.5 s is 1.487 rad at t = 3.38 s, distance 3.283 m, then the commands return to the flat offset (about -0.27, -0.41, 0.03, -0.18 rad at t = 5 s). Speed never falls below 0.5 m/s (minimum after 0.5 s is 0.746 m/s). The PD baseline also keeps the gate off, crosses 1 rad/s of knee rate at 1.390 m, and ends at 5.492 m by pitching. This clip is that pitch. The knees move with the body and release. They do not wind. Clearance passes. It is not an earlier latched tumble than a PD that never opens the gate.

Frames: at 1 s the body is level on the blue steps. At 2 s and 3 s it is pitched nose-down on the stair face, wheels on the treads. At 5 s it is level on the gray surround with the stairs behind it.

## Inverted stairs, hard

Directory: `cycle2_checkpoint_inverted-stairs_hard`.

Knee rate now rises before the stall, and earlier in distance than the PD gate. The cell is still not exited. Final distance 6.835 m is past the PD's 2.929 m and past cycle 1's 6.665 m. The end state is not a cruise: speed 0.839 m/s, knee rate 2.378 rad/s, uprightness 0.752, scan 0.613 m, gate 0. Uprightness minimum is 0.706 at t = 4.16 s. End knee commands are about 0.94, 2.41, -0.87, 0.70 rad. Peak absolute knee command after 0.5 s is 8.022 rad at t = 2.04 s, distance 1.485 m, not the PD's windup to about 19 rad. Mean action MSE against the PD is 5.852.

Knee-rate onset after 0.5 s is t = 0.84 s, distance 0.832 m, speed 1.050 m/s, knee rate 1.108 rad/s, uprightness 1.000, scan 0.229 m, gate still 0. Speed first falls below 0.5 m/s at t = 0.92 s, distance 0.906 m, while knee rate is already 1.150 rad/s. The PD baseline opens the gate at 0.909 m after speed has already fallen. This onset is earlier than 0.909 m, and it is before the stall. Cycle 1's onset was 0.989 m at speed -0.175 m/s. The teacher gate inside this rollout first opens at 1.335 m, later than the baseline gate, and is not held for the rest of the clip (it is off at t = 8 s).

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.471 | 1.097 | 0.270 | 1.000 | 0.220 | 0 |
| 0.84 | 0.832 | 1.050 | 1.108 | 1.000 | 0.229 | 0 |
| 1.00 | 0.932 | -0.001 | 2.257 | 0.998 | 0.265 | 0 |
| 2.00 | 1.458 | 0.595 | 3.958 | 0.863 | 0.291 | 1 |
| 5.00 | 3.675 | 1.051 | 2.673 | 0.773 | 0.358 | 0 |
| 7.00 | 5.863 | 1.052 | 4.236 | 0.798 | 0.285 | 0 |
| 8.00 | 6.835 | 0.839 | 2.378 | 0.752 | 0.613 | 0 |

Frames: at 1 s the body is level on the red pit floor with the stair faces ahead and the legs not folded. At 2 s it is pitched on the red stair face. At 5 s and 7 s it is still on the red stairs, tilted, short of the gray surround. The cell is not cleared.

## Box grid, hard

Directory: `cycle2_checkpoint_box-grid_hard`.

This clip clears, and the knee-rate rise starts before the PD's gate distance. End: distance 6.634 m, speed 1.087 m/s, knee rate 0.404 rad/s, uprightness 0.999, scan 0.253 m, gate 0. The PD baseline ends at 1.603 m, uprightness 0.664, gate 1. Cycle 1 ended at 7.974 m on the same cruise. Uprightness minimum here is 0.935 at t = 4.32 s. Mean action MSE against the PD is 5.160. Peak absolute knee command is 5.968 rad at t = 5.62 s, distance 3.925 m. End knee commands are about -0.25, -0.37, -0.03, -0.11 rad, the flat offset. They do not wind to about 19 rad.

Knee-rate onset after 0.5 s is t = 0.54 s, distance 0.431 m, speed 0.870 m/s, knee rate 1.193 rad/s, uprightness 0.998, scan 0.413 m, gate 0. Forward scan height starts near 0.40 m, so the cubes are already in the scan. Speed does not fall below 0.5 m/s until t = 1.48 s, distance 1.425 m. The PD baseline opens the gate at 0.903 m with speed already 0.256 m/s. The student's knee-rate onset is earlier than that gate, and speed has not fallen yet. Cycle 1's onset was 0.542 m. The teacher gate inside this rollout opens later, at 1.967 m, and is off again by the end (155 frames on, last at 4.12 m).

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.54 | 0.431 | 0.870 | 1.193 | 0.998 | 0.413 | 0 |
| 1.00 | 0.888 | 1.143 | 2.054 | 0.964 | 0.470 | 0 |
| 2.00 | 1.851 | 1.073 | 3.691 | 0.988 | 0.287 | 0 |
| 5.00 | 3.597 | 0.366 | 2.410 | 0.962 | 0.367 | 1 |
| 7.00 | 5.536 | 1.221 | 3.322 | 0.998 | 0.276 | 0 |
| 8.00 | 6.634 | 1.087 | 0.404 | 0.999 | 0.253 | 0 |

Frames: at 1 s the body is tilted among the green cubes, wheels down. At 2 s it is level in the grid. At 5 s one side is lifted in the grid, not on its side. At 7 s it is level on the gray plane with the grid behind it. Knee rate is still 3.322 rad/s at that 7 s frame and has fallen to 0.404 rad/s by 8 s, with the knee commands back at the flat offset.

Clearance is at least the PD's, and knee-rate onset is earlier in distance than the PD gate. The motion releases. It is not the latched windup.

## Scan ablation, pyramid stairs hard

Directory: `cycle2_checkpoint_pyramid-stairs_hard_ablate-scan`.

Zeroing `height_scan` changes the clip. The gate stays 0 in both conditions. With the scan, knee-rate onset after 0.5 s is 1.235 m. Without it, knee rate stays under 1 rad/s until t = 3.30 s, distance 4.504 m, speed 0.504 m/s, knee rate 1.079 rad/s, uprightness 0.956, scan 0.324 m. Uprightness first falls below 0.98 at t = 1.08 s, distance 1.346 m, speed 1.983 m/s, knee rate 0.509 rad/s, scan 0.764 m. With the scan that uprightness drop is at 1.282 m with knee rate already 1.137 rad/s. End of the ablated clip: distance 10.399 m, speed 1.248 m/s, knee rate 0.018 rad/s, uprightness 0.998, scan 0.310 m, gate 0. Uprightness minimum is 0.707 at t = 2.26 s. With the scan the clip ends at 8.345 m and 1.085 m/s. Mean action MSE on the ablated clip is 0.710, against 0.150 with the scan.

Knee-rate onset after 0.5 s therefore changes from 1.235 m to 4.504 m. Final distance changes from 8.345 m to 10.399 m. Cycle 1 moved the same onset from 1.366 m to never, and distance from 8.662 m to 11.920 m. The ablation does not leave the clip unchanged. Both gates stay off. The early pitch distance is almost the same (1.282 m with the scan, 1.346 m without), and the ablated body is already going faster when it pitches. The scan changes when the knees cross 1 rad/s.

Frames, ablated: at 1 s the body is already pitched nose-down on the blue steps. At 2 s it is nose-down near the bottom of the stairs. At 3 s it is on the lower landing, still pitched. At 7 s it is level on the gray plane. The with-scan frame at 1 s is still level.

## Hacking signs

- Knee spin on flat: no. After 1 s, knee-rate means are 0.044 rad/s on both difficulties. Hard stays under 0.099 rad/s. Medium has one frame at 0.107 rad/s. The startup spike is under 1 rad/s by 0.20 s. Knee commands stay under half a radian.
- High teacher match on stairs with no clearance: no on the cells that clear. Pyramid action MSE is 0.150 and the clip ends on the flat cruise at 8.345 m. Box-grid MSE is 5.160 and the clip ends upright on the gray surround at 6.634 m. Inverted MSE is 5.852 and that cell is not cleared, so it is not a quiet copy of the PD either (the PD winds to about 19 rad; these commands peak near 8 rad).
- Spinning in place: no. Flat ends near 8.68 m. The stair and grid clips all travel several meters.
- Falling or bouncing for speed: no on flat (uprightness stays at 0.999). The box grid dips to 0.935 and recovers onto the gray plane. Inverted stairs stay tilted (minimum 0.706, end 0.752) and do not buy a flat cruise that way.
- Scan ablation leaves the clip unchanged: no. Knee-rate onset moves from 1.235 m to 4.504 m, and final distance moves from 8.345 m to 10.399 m.

## Goal checks

Compared with the PD numbers in `BASELINE_REVIEW.md` and with cycle 1 in `CYCLE1_REVIEW.md`. One seed only. Flat is scored against the 1 m/s command. The PD's 0.697 m/s cruise is reported as a gap, and it is not the fail rule.

- **Flat, medium and hard: knee rate near zero, speed near the 1 m/s command. Pass.** After 1 s the knee-rate means are 0.044 and 0.044 rad/s. Hard stays under 0.1 rad/s. Medium has one frame at 0.107 rad/s. Speed means are 1.093 and 1.094 m/s. Command error is about +0.09 m/s. The gap versus the PD mean of 0.697 m/s is about +0.40 m/s. That gap is not a failure.
- **Hard inverted stairs and hard box grid: clearance at least the PD, and knee-rate onset earlier in distance than the PD gate. Fail.** Box grid passes both parts (6.634 m and an upright exit onto the gray surround, against the PD's 1.603 m tipped; knee-rate onset at 0.431 m against the PD gate at 0.903 m). Inverted onset passes: knee rate crosses 1 rad/s at 0.832 m while speed is still 1.050 m/s, earlier than the PD gate at 0.909 m and earlier than cycle 1's 0.989 m. Inverted clearance fails. Distance 6.835 m is past the PD's 2.929 m, and the 5 s and 7 s frames are still on the red stairs, short of the gray surround. Uprightness at 8 s is 0.752.
- **Hard pyramid stairs: cell cleared, and the motion is a pitch rather than a tumble. Pass.** End distance 8.345 m, speed 1.085 m/s, knee rate 0.039 rad/s, uprightness 0.999, gate 0, against the PD's 5.492 m pitched cruise. The gate never opens. Knee commands peak near 1.5 rad and return. Frames go from level on the steps, to nose-down on the stair face, to level on the gray surround. The knees move with that pitch. They do not latch.
- **Scan ablation on hard pyramid stairs changes tumble onset. Pass.** Knee-rate onset after 0.5 s is 1.235 m with the scan and 4.504 m without it. Final distance is 8.345 m with the scan and 10.399 m without it. The with-scan frame at 1 s is level. The ablated frame at 1 s is already pitched. This is not a latched tumble turning on and off. The onset still moves, so the clip is not scan-blind.
- **Two seeds agree. Fail.** Only seed 42 was trained. Two-seed agreement is not met.

The paper goal is not passed. The study is unfinished. Reward at iteration 998 is 53.26, against 48.61 at the cycle-1 checkpoint, and that does not overturn the failed checks.
