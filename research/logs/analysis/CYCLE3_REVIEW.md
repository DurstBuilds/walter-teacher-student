# Cycle 3 review

**Date:** 2026-10-14
**Checkpoint:** `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_01-31-29/model_1497.pt` (checkpoint field `iter` is 1497). Resume loaded `model_998.pt` at iteration 998 and ran 500 additional updates, iterations 998 through 1497.
**Seed:** 42. `scripts/train.py` still sets `cfg.seed = 42`. No second seed.
**What changed:** `predictive_knee_tumble` still pays when measured speed along the command is at or below zero, if the forward command is large enough and the scan gate is open. The scan step is now the largest absolute jump, so a drop opens the gate as well as a rise. The 0.04 m to 0.08 m ramp was not changed. Scan size, scan offset, and the terrain mix were not changed. Residual actions were not added. No flat-speed penalty was added.

**Train command, exit 0:**

`uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500 --resume research/logs/training/walter_fs_teacher_rough-test/2026-10-14_00-33-54/model_998.pt`

Logs: `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_01-31-29/`. 2048 environments, CUDA.

**Clips:** `uv run python scripts/eval_cells.py --controller checkpoint --checkpoint research/logs/training/walter_fs_teacher_rough-test/2026-10-14_01-31-29/model_1497.pt` with `--output` set per cell. Default 8 s, default 1 m/s command, eval seed 0. PD baselines were not rerecorded. PD numbers below are quoted from `research/logs/analysis/BASELINE_REVIEW.md`. Cycle 2 numbers are quoted from `research/logs/analysis/CYCLE2_REVIEW.md`.

| Clip | Directory |
| --- | --- |
| flat medium | `research/logs/analysis/cycle3_checkpoint_flat_medium` |
| flat hard | `research/logs/analysis/cycle3_checkpoint_flat_hard` |
| pyramid stairs hard | `research/logs/analysis/cycle3_checkpoint_pyramid-stairs_hard` |
| inverted stairs hard | `research/logs/analysis/cycle3_checkpoint_inverted-stairs_hard` |
| box grid hard | `research/logs/analysis/cycle3_checkpoint_box-grid_hard` |
| pyramid stairs hard, scan ablated | `research/logs/analysis/cycle3_checkpoint_pyramid-stairs_hard_ablate-scan` |

Frames below were taken from each `clip.mp4`. The numbers are from `frames.csv`.

**Onset used in the checks.** Same rule as cycles 1 and 2. The PD reference is the baseline gate distance. Student knee-rate onset is the first frame after 0.5 s with `knee_rate` above 1 rad/s. The first half-second is a startup transient.

## Why the term changed

Cycle 2 left the reward unchanged and the hard inverted cell still did not clear. The code at that checkpoint multiplied knee rate by `relu(body_vx)` and by the scan gate, and the step measure kept only upward jumps.

The cycle-2 inverted-stair csv does not show speed going negative before the knees spin. Knee rate crosses 1 rad/s at t = 0.84 s, distance 0.832 m, speed 1.050 m/s, scan 0.229 m. A step is already in the scan before that stall: at t = 0.40 s, distance 0.361 m, speed 1.150 m/s, knee rate 0.218 rad/s, forward-scan height is 0.222 m. The same run's flat cruise ends at 0.255 m. Less air is higher terrain ahead. That depression stays near 0.220 m through t = 0.76 s while speed stays near 1.1 m/s.

Speed does then go to or below zero while the robot is still at the stair and the knees are already spinning. The first such frame after startup is t = 1.00 s, distance 0.932 m, speed -0.001 m/s, knee rate 2.257 rad/s, scan 0.265 m. Speed bottoms at t = 1.76 s, -0.773 m/s, distance 1.363 m, knee rate 4.605 rad/s, scan 0.300 m. The 2 s frame of that clip is on the red stair face, not past it. `relu(body_vx)` is zero on those frames, so `predictive_knee_tumble` paid nothing for the tumble that was already happening, and the only knee term left was the reactive lag term.

The change is therefore not "pay before the first spin." The first spin in cycle 2 was already at positive speed. The change is that a forward command plus an open scan gate still pays for knee rate when measured forward speed is at or below zero. Positive speed is still clamped at 1, so a moving robot is scaled as before. A shut gate does not get that stall credit, and the return is still multiplied by the gate, so a flat scan pays nothing.

The gate still ignored drops. Cycle 2's probe of hard pyramid stairs read step 0.000 m because `_max_occupied_rise` applied `relu`. The step is now the largest absolute jump between consecutive occupied bins. The match gate is still 1 at or below 0.04 m and 0 at or above 0.08 m.

An import-level check of those functions, with synthetic forward bins, gave: flat profile step 0.000 m, match 1, tumble 0; a +0.180 m rise and a -0.180 m drop both step 0.180 m, match 0, tumble 1; a ±0.045 m slope steps 0.045 m, match 0.875, tumble 0.125; a 0.060 m step is match 0.500. Stall progress is 1.0 at speed 0 and at speed -0.5 when the tumble gate is 1, stays 0.4 at speed 0.4, and stays 0 when the gate is shut at speed 0 or -0.2.

Residual actions were not added. On the cycle-2 inverted clip the approach still had positive speed (1.050 m/s at the knee-rate crossing) and a depressed scan (0.229 m against a flat 0.255 m), and the cycle-2 probe already had the tumble gate open on that cell. The policy was not a copy of the PD there: action MSE was 5.852, and the peak knee command was 8.022 rad rather than the PD windup near 19 rad.

## Reward curve

Scalars are from this run's TensorBoard file. Iteration 998 in this log is the first resumed update after `init_at_random_ep_len` restarts the episodes, so its mean reward (2.33) and episode length (39.6) are incomplete episodes, not the cycle-2 checkpoint's 53.26. By iteration 1048 the episode statistics are back in that band. Mean reward then stays there. It does not decide the checks. `predictive_knee_tumble` stays small next to linear-velocity tracking. It is a bit larger than cycle 2's 0.089 at iteration 998.

| Iteration | Mean reward | Episode length | `track_teacher` | `predictive_knee_tumble` | `knee_stuck_tumble` | `track_linear_velocity` | Teacher action MSE | Terrain level mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1048 | 52.12 | 947.0 | 0.459 | 0.117 | 0.046 | 1.537 | 2.54 | 2.28 |
| 1248 | 54.61 | 931.8 | 0.600 | 0.092 | 0.044 | 1.595 | 2.63 | 2.15 |
| 1398 | 53.82 | 949.8 | 0.536 | 0.111 | 0.043 | 1.542 | 2.48 | 2.12 |
| 1497 | 51.90 | 920.8 | 0.501 | 0.129 | 0.063 | 1.509 | 4.30 | 2.12 |

Iteration 1496 mean reward was 53.61, so the last point is inside the same band as the late run, not a collapse. Terrain level mean ends at 2.12 of a max of 3. Cycle 2 ended at reward 53.26 and terrain level 2.15.

## Flat

Directories: `cycle3_checkpoint_flat_medium`, `cycle3_checkpoint_flat_hard`.

Both difficulties are flat, and the traces agree. The PD tumble gate is 0 on every frame. Uprightness minimum is 1.000. After the first second, knee rate stays near zero.

Medium, after the first second: forward speed stays between 0.994 and 1.070 m/s (mean 1.027). Knee rate stays at or below 0.058 rad/s (mean 0.016). At t = 8.00 s, distance is 8.190 m, forward speed is 1.018 m/s, knee rate is 0.018 rad/s, uprightness is 1.000, forward-scan height is 0.264 m, and the tumble gate is 0. Knee-rate onset after 0.5 s does not occur. The startup spike peaks at 1.566 rad/s at t = 0.06 s and is under 1 rad/s by t = 0.12 s. End knee commands are about -0.25, -0.30, 0.60, 0.54 rad. They do not wind. Peak absolute knee command after 0.5 s is 0.662 rad. Mean action MSE against the PD on this clip is 0.092.

Hard matches it. After the first second, speed stays between 0.995 and 1.075 m/s (mean 1.027) and knee rate stays at or below 0.057 rad/s (mean 0.014). End: distance 8.187 m, speed 1.034 m/s, knee rate 0.010 rad/s, uprightness 1.000, scan 0.264 m, gate 0. End knee commands are about -0.21, -0.30, 0.60, 0.55 rad. Mean action MSE is 0.092.

Command error is about +0.027 m/s on both difficulties (mean 1.027 against the 1.0 m/s command). The gap versus the PD cruise (mean 0.697 m/s in `BASELINE_REVIEW.md`) is about +0.33 m/s. Cycle 2 was a mean of 1.093 and 1.094 m/s with knee-rate means 0.044 rad/s. This run is closer to the command and the knees are quieter. The extra speed against the PD is the PD's undershoot, not a flat failure. The head-knee targets near 0.6 rad are a held pose. The rate on those joints stays near zero.

Frames at 1 s, 3 s, and 7 s on hard, and at 1 s and 7 s on medium, show the body level, wheels on the plane, and no folded legs. Nothing in the frames looks like spinning in place or bouncing.

## Pyramid stairs, hard

Directory: `cycle3_checkpoint_pyramid-stairs_hard`.

The PD tumble gate stays 0 for all 400 frames. The robot pitches over the steps and is back on the flat cruise by the end. Final distance 7.957 m is past the PD's 5.492 m and short of cycle 2's 8.345 m. End state is the student's flat cruise: speed 1.006 m/s, knee rate 0.027 rad/s, uprightness 1.000, scan 0.265 m, gate 0. Uprightness minimum is 0.862 at t = 1.84 s. Mean action MSE against the PD is 0.273. Speed after 0.5 s stays at or above 0.640 m/s.

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.76 | 0.680 | 0.899 | 1.057 | 0.998 | 0.431 | 0 |
| 1.00 | 0.904 | 0.887 | 0.940 | 0.994 | 0.533 | 0 |
| 2.00 | 1.937 | 1.373 | 1.093 | 0.943 | 0.804 | 0 |
| 2.36 | 2.297 | 0.826 | 1.342 | 0.919 | 0.676 | 0 |
| 5.00 | 4.876 | 1.004 | 0.015 | 1.000 | 0.265 | 0 |
| 8.00 | 7.957 | 1.006 | 0.027 | 1.000 | 0.265 | 0 |

Knee rate next exceeds 1 rad/s after the startup window at t = 0.76 s, distance 0.680 m, speed 0.899 m/s, scan 0.431 m, gate 0. Scan height there is already above the flat 0.264 m, so the forward rays see more air. That is the drop this cycle's gate can open. Cycle 2's crossing on the same cell was at 1.235 m with the upward-only gate. The knee-rate bursts last through t = 3.84 s, distance 3.655 m, and knee rate is back under 0.1 rad/s at t = 4.52 s, distance 4.379 m. Peak absolute knee command after 0.5 s is 2.019 rad at t = 2.36 s, distance 2.297 m, then the commands return to the flat offset (about -0.29, -0.31, 0.59, 0.53 rad at t = 5 s). They do not wind.

Frames: at 1 s the body is level on the blue steps. At 2 s and 3 s it is pitched nose-down on the stair face, wheels on the treads. At 5 s it is level on the gray surround with the stairs behind it. The knees bend with that pitch and release. This is not a latched tumble.

## Inverted stairs, hard

Directory: `cycle3_checkpoint_inverted-stairs_hard`.

Knee rate rises before the stall, and earlier in distance than the PD gate. The cell is still not exited. Final distance 6.820 m is past the PD's 2.929 m and in the same place as cycle 2's 6.835 m. The end state is not a cruise: speed 0.823 m/s, knee rate 1.401 rad/s, uprightness 0.826, scan 0.324 m, gate 0. Uprightness minimum is 0.710 at t = 7.06 s. End knee commands are about 1.90, 2.18, 1.66, 2.40 rad. Peak absolute knee command after 0.5 s is 10.548 rad at t = 1.78 s, distance 1.423 m, against cycle 2's 8.022 rad and the PD's windup to about 19 rad. Mean action MSE against the PD is 4.704.

Knee-rate onset after 0.5 s is t = 0.74 s, distance 0.718 m, speed 1.008 m/s, knee rate 1.092 rad/s, uprightness 1.000, scan 0.237 m, gate still 0. Speed first falls below 0.5 m/s at t = 0.80 s, distance 0.765 m, while knee rate is already 2.511 rad/s. Speed first goes to or below zero at t = 0.90 s, distance 0.793 m, speed -0.248 m/s, knee rate 3.419 rad/s, scan 0.271 m. The PD baseline opens the gate at 0.909 m after speed has already fallen. This onset is earlier than 0.909 m, and it is before the stall. Cycle 2's onset was 0.832 m at speed 1.050 m/s. The PD gate inside this rollout first opens at 1.305 m, later than the baseline gate, and is off again by t = 8 s (102 frames on, last at 3.640 m).

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 0.469 | 1.052 | 0.377 | 1.000 | 0.232 | 0 |
| 0.74 | 0.718 | 1.008 | 1.092 | 1.000 | 0.237 | 0 |
| 0.90 | 0.793 | -0.248 | 3.419 | 0.996 | 0.271 | 0 |
| 2.00 | 1.518 | 0.761 | 3.524 | 0.936 | 0.319 | 1 |
| 5.00 | 3.902 | 0.931 | 3.603 | 0.755 | 0.287 | 0 |
| 7.00 | 5.790 | 0.940 | 2.842 | 0.726 | 0.342 | 0 |
| 8.00 | 6.820 | 0.823 | 1.401 | 0.826 | 0.324 | 0 |

Frames: at 1 s the body is at the red stair face with the legs starting to fold. At 2 s it is pitched on the red stairs with one leg lifted. At 5 s and 7 s it is still on the red stairs, tilted, short of the gray surround. The cell is not cleared. The motion is a tumble attempt, not a pitch that releases: knee rate stays above 1 rad/s through the end, and the knee commands do not return to the flat offset.

## Box grid, hard

Directory: `cycle3_checkpoint_box-grid_hard`.

This clip does not clear. End: distance 5.320 m, speed -0.186 m/s, knee rate 2.477 rad/s, uprightness 0.997, scan 0.318 m, gate 1. The PD baseline ends at 1.603 m, uprightness 0.664, gate 1. Cycle 2 ended at 6.634 m, speed 1.087 m/s, knee rate 0.404 rad/s, uprightness 0.999, gate 0, on the gray surround. Uprightness minimum here is 0.930 at t = 5.74 s. Mean action MSE against the PD is 6.941. Peak absolute knee command is 9.603 rad at t = 7.62 s, distance 5.241 m. End knee commands are about 2.13, 3.29, 4.82, 5.74 rad. They do not return to the flat offset, and they do not reach the PD's windup near 19 rad.

Knee rate is already 1.783 rad/s at t = 0.30 s, distance 0.278 m, and 4.148 rad/s at t = 0.48 s, distance 0.425 m. Knee-rate onset after 0.5 s is t = 0.52 s, distance 0.450 m, speed 0.507 m/s, knee rate 4.191 rad/s, uprightness 0.969, scan 0.514 m, gate 0. It dips to 0.741 rad/s at t = 0.80 s and is 4.507 rad/s again at t = 1.00 s, distance 0.799 m. Forward scan height starts near 0.42 m, so the cubes are already in the scan. Speed falls below 0.5 m/s at t = 0.54 s, distance 0.462 m, while knee rate is 4.241 rad/s. The PD baseline opens the gate at 0.903 m with speed already 0.256 m/s. The student's knee-rate onset is earlier than that gate. Cycle 2's onset was 0.431 m, and that clip then released. This one does not. The PD gate inside this rollout opens at 2.646 m and is still on at t = 8 s (174 frames on).

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.52 | 0.450 | 0.507 | 4.191 | 0.969 | 0.514 | 0 |
| 1.00 | 0.799 | 1.056 | 4.507 | 0.980 | 0.517 | 0 |
| 2.00 | 1.617 | 0.912 | 3.554 | 0.960 | 0.317 | 0 |
| 5.00 | 3.432 | 0.223 | 3.014 | 0.983 | 0.394 | 1 |
| 7.00 | 4.828 | 1.355 | 3.494 | 0.995 | 0.321 | 1 |
| 8.00 | 5.320 | -0.186 | 2.477 | 0.997 | 0.318 | 1 |

Frames: at 1 s the body is in the green cubes with the legs folded. At 2 s it is pitched in the grid, wheels down. At 5 s and 7 s it is still in the green grid, legs folded, short of the gray surround. Clearance is not the cycle-2 exit. Distance 5.320 m is past the PD's 1.603 m, and the body stays upright, but the 7 s frame is still in the obstacle and the end state is a latched tumble (gate 1, speed negative, knee commands still several radians). Pitching over without a tumble is not what this clip is. It tumbles and does not leave.

## Scan ablation, pyramid stairs hard

Directory: `cycle3_checkpoint_pyramid-stairs_hard_ablate-scan`.

Zeroing `height_scan` changes the clip. The PD gate stays 0 in both conditions. With the scan, knee-rate onset after 0.5 s is 0.680 m. Without it, knee rate stays under 1 rad/s after 0.5 s. The maximum after 0.5 s is 0.817 rad/s at t = 1.54 s, distance 2.693 m, speed 1.872 m/s, uprightness 0.827, scan 0.704 m. Uprightness first falls below 0.98 at t = 0.80 s, distance 1.273 m, speed 2.122 m/s, knee rate 0.219 rad/s, scan 0.775 m. With the scan that uprightness drop is at 1.125 m with knee rate 0.598 rad/s. End of the ablated clip: distance 12.800 m, speed 1.613 m/s, knee rate 0.000 rad/s, uprightness 0.998, scan 0.640 m, gate 0. Uprightness minimum is 0.768 at t = 1.76 s. Scan height is 0.326 m at t = 7.00 s and rises only in the last frames (0.438 m at t = 7.96 s, 0.640 m at t = 8.00 s) as distance passes 12.7 m. With the scan the clip ends at 7.957 m and 1.006 m/s. Mean action MSE on the ablated clip is 0.668, against 0.273 with the scan.

Knee-rate onset after 0.5 s therefore changes from 0.680 m to never. Final distance changes from 7.957 m to 12.800 m. Cycle 2 moved the same onset from 1.235 m to 4.504 m, and distance from 8.345 m to 10.399 m. The ablation does not leave the clip unchanged. Both PD gates stay off. The with-scan frame at 1 s is still level on the steps. The ablated frame at 1 s is already pitched nose-down. At 3 s and 7 s the ablated body is level on the gray plane.

## Hacking signs

- Knee spin on flat: no. After 1 s, knee-rate means are 0.016 rad/s (medium) and 0.014 rad/s (hard). Every frame after 1 s is at or below 0.058 rad/s. The startup spike is under 1 rad/s by 0.12 s. Knee commands stay under 0.67 rad.
- High teacher match on stairs with no clearance: no on the cell that clears. Pyramid action MSE is 0.273 and the clip ends on the flat cruise at 7.957 m. Inverted MSE is 4.704 and box-grid MSE is 6.941. Those two cells are not cleared, and they are not quiet copies of the PD either (the PD winds to about 19 rad; these commands peak at 10.548 rad and 9.603 rad).
- Spinning in place: no. Flat ends near 8.19 m. The stair and grid clips all travel several meters.
- Falling or bouncing for speed: no on flat (uprightness stays at 1.000). The box grid dips to 0.930 and stays upright inside the grid. Inverted stairs stay tilted (minimum 0.710, end 0.826) and do not buy a flat cruise that way.
- Scan ablation leaves the clip unchanged: no. Knee-rate onset moves from 0.680 m to never, and final distance moves from 7.957 m to 12.800 m.

## Goal checks

Compared with the PD numbers in `BASELINE_REVIEW.md` and with cycle 2 in `CYCLE2_REVIEW.md`. One seed only. Flat is scored against the 1 m/s command. The PD's 0.697 m/s cruise is reported as a gap, and it is not the fail rule. Clearance means the cell is left, which is the standard used when cycle 2 failed inverted stairs despite a longer distance than the PD.

- **Flat, medium and hard: knee rate near zero, speed near the 1 m/s command. Pass.** After 1 s the knee-rate means are 0.016 and 0.014 rad/s, and every frame is at or below 0.058 rad/s. Speed means are 1.027 m/s on both difficulties. Command error is about +0.027 m/s. The gap versus the PD mean of 0.697 m/s is about +0.33 m/s. That gap is not a failure.
- **Hard inverted stairs and hard box grid: clearance at least the PD, and knee-rate onset earlier in distance than the PD gate. Fail.** Inverted onset passes: knee rate crosses 1 rad/s at 0.718 m while speed is still 1.008 m/s, earlier than the PD gate at 0.909 m and earlier than cycle 2's 0.832 m. Inverted clearance fails. Distance 6.820 m is past the PD's 2.929 m, and the 5 s and 7 s frames are still on the red stairs. Uprightness at 8 s is 0.826. Box onset passes: 0.450 m against the PD gate at 0.903 m. Box clearance fails. Distance 5.320 m is past the PD's 1.603 m, but the 5 s and 7 s frames are still in the green grid, end speed is -0.186 m/s, the PD gate is still 1, and the knee commands are still several radians. Cycle 2 had left that cell upright on the gray surround at 6.634 m.
- **Hard pyramid stairs: cell cleared, and the motion is a pitch rather than a tumble. Pass.** End distance 7.957 m, speed 1.006 m/s, knee rate 0.027 rad/s, uprightness 1.000, gate 0, against the PD's 5.492 m pitched cruise. The PD gate never opens. Knee commands peak at 2.019 rad and return. Frames go from level on the steps, to nose-down on the stair face, to level on the gray surround. The knees move with that pitch. They do not latch.
- **Scan ablation on hard pyramid stairs changes tumble onset. Pass.** Knee-rate onset after 0.5 s is 0.680 m with the scan and does not occur without it (maximum after 0.5 s is 0.817 rad/s). Final distance is 7.957 m with the scan and 12.800 m without it. The with-scan frame at 1 s is level. The ablated frame at 1 s is already pitched.
- **Two seeds agree. Fail.** Only seed 42 was trained. Two-seed agreement is not met.

The paper goal is not passed. The study is unfinished. Reward at iteration 1497 is 51.90, against 53.26 at the cycle-2 checkpoint, and that does not overturn the failed checks.
