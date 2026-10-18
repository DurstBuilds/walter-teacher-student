# Cycle 1 review

**Date:** 2026-10-13
**Checkpoint:** `research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-32-02/model_499.pt` (checkpoint field `iter` is 499). This is the last file of a 500-iteration run (iterations 0 through 499).
**Seed:** 42. `scripts/train.py` sets `cfg.seed = 42`, and the PPO runner config also uses seed 42. No `--seed` flag was added.
**What ran:** 2048 environments, 500 iterations, CUDA. The 4096-environment attempt died during scene creation (`Failed to allocate 6366953472 bytes on device 'cuda:0'`) and was not resumed. Reward terms were not changed. No residual actions, no scan-size change, no terrain-mix change.

**Train commands:**

- Smoke, exit 0: `uv run python scripts/train.py --teacher --task rough-test --num-envs 8 --max-iterations 2`
- OOM, not the evaluated run: `uv run python scripts/train.py --teacher --task rough-test --num-envs 4096 --max-iterations 500`
- Evaluated run, exit 0: `uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500`

Logs for the evaluated run are `research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-32-02/`. The smoke log is the earlier directory `2026-10-13_23-30-59` and is not this checkpoint.

**Clips:** `uv run python scripts/eval_cells.py --controller checkpoint --checkpoint research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-32-02/model_499.pt` with `--output` set per cell. Default 8 s, default 1 m/s command, eval seed 0. PD baselines were not rerecorded. PD numbers below are quoted from `research/logs/analysis/BASELINE_REVIEW.md`.

| Clip | Directory |
| --- | --- |
| flat medium | `research/logs/analysis/cycle1_checkpoint_flat_medium` |
| flat hard | `research/logs/analysis/cycle1_checkpoint_flat_hard` |
| pyramid stairs hard | `research/logs/analysis/cycle1_checkpoint_pyramid-stairs_hard` |
| inverted stairs hard | `research/logs/analysis/cycle1_checkpoint_inverted-stairs_hard` |
| box grid hard | `research/logs/analysis/cycle1_checkpoint_box-grid_hard` |
| pyramid stairs hard, scan ablated | `research/logs/analysis/cycle1_checkpoint_pyramid-stairs_hard_ablate-scan` |

Frames below were taken from each `clip.mp4`. The numbers are from `frames.csv`.

**Onset used in the checks.** The PD reference is the baseline gate distance, not the teacher gate inside the student rollout (that gate moves when the body moves differently). Student knee-rate onset is the first frame after 0.5 s with `knee_rate` above 1 rad/s. The first half-second is a startup transient on every clip, including flat, and is gone as a tumble before 1 s. A latched tumble, in the baseline, winds the knee targets to about 19 rad and leaves the gate on. A pitch, in the baseline pyramid clip, spikes knee rate and then returns, with the gate still off.

## Reward curve

Scalars are from this run's TensorBoard file. Mean reward rose. It does not decide the checks. `predictive_knee_tumble` stays small next to linear-velocity tracking.

| Iteration | Mean reward | Episode length | `track_teacher` | `predictive_knee_tumble` | `knee_stuck_tumble` | `track_linear_velocity` | Teacher action MSE | Terrain level mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.58 | 39.2 | 0.013 | 0.000 | 0.005 | 0.006 | 1.29 | 1.28 |
| 50 | 30.93 | 932.7 | 0.290 | 0.019 | 0.080 | 1.053 | 50.64 | 2.33 |
| 100 | 37.75 | 954.8 | 0.344 | 0.022 | 0.068 | 1.227 | 34.19 | 2.13 |
| 250 | 46.13 | 922.6 | 0.442 | 0.038 | 0.048 | 1.420 | 6.56 | 2.16 |
| 499 | 48.61 | 915.7 | 0.486 | 0.066 | 0.057 | 1.437 | 5.41 | 2.11 |

Iteration 497 mean reward was 52.52 and iteration 498 was 50.87, so the last point is inside the same band as the late run, not a collapse. Terrain level mean ends at 2.11 of a max of 3.

## Flat

Directories: `cycle1_checkpoint_flat_medium`, `cycle1_checkpoint_flat_hard`.

Both difficulties are flat, and the traces agree. The gate is 0 on every frame. Uprightness minimum is 0.9999. After the first second, every frame has knee rate under 0.1 rad/s.

Medium, after the first second: forward speed stays between 1.100 and 1.150 m/s (mean 1.132). Knee rate stays under 0.082 rad/s (mean 0.025). At t = 8.00 s, distance is 8.976 m, forward speed is 1.100 m/s, knee rate is 0.082 rad/s, uprightness is 1.000, forward-scan height is 0.250 m, and the tumble gate is 0. Knee-rate onset after 0.5 s does not occur. The startup spike peaks at 2.251 rad/s at t = 0.06 s and is under 1 rad/s by t = 0.18 s. End knee commands are about -0.24, -0.43, -0.49, -0.42 rad. They do not wind. Mean action MSE against the PD on this clip is 0.129.

Hard matches it. After the first second, speed stays between 1.100 and 1.148 m/s (mean 1.132) and knee rate stays under 0.052 rad/s (mean 0.024). End: distance 8.975 m, speed 1.121 m/s, knee rate 0.034 rad/s, uprightness 1.000, scan 0.250 m, gate 0.

The PD flat cruise in `BASELINE_REVIEW.md` is 0.692–0.700 m/s (mean 0.697), knee rate under 0.017 rad/s (mean 0.0005), distance 5.536 m, scan 0.267 m, gate 0. Student knee rate is near zero. Student speed is about 0.43 m/s faster than that cruise, and it sits above the 1 m/s command. That is not the PD band.

Frames at 1 s, 3 s, and 7 s show the body level, wheels on the plane, and no folded legs. Nothing in the frames looks like spinning in place or bouncing. The held knee targets near -0.4 rad are a pose offset from the PD's near-zero targets, not a spin.

## Pyramid stairs, hard

Directory: `cycle1_checkpoint_pyramid-stairs_hard`.

The tumble gate stays 0 for all 400 frames. The robot pitches over the steps and is back on the flat cruise by the end. Final distance 8.662 m is past the PD's 5.492 m. End state is the student's flat cruise: speed 1.142 m/s, knee rate 0.031 rad/s, uprightness 1.000, scan 0.251 m, gate 0. Uprightness minimum is 0.853 at t = 2.72 s. Mean action MSE against the PD is 0.199.

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.64 | 0.605 | 1.061 | 0.259 | 1.000 | 0.424 | 0 |
| 1.00 | 0.983 | 0.949 | 0.359 | 1.000 | 0.561 | 0 |
| 1.40 | 1.366 | 1.095 | 1.121 | 0.979 | 0.717 | 0 |
| 2.50 | 2.466 | 1.214 | 1.356 | 0.909 | 0.648 | 0 |
| 3.16 | 3.148 | 1.058 | 2.158 | 0.958 | 0.358 | 0 |
| 5.00 | 5.266 | 1.135 | 0.029 | 1.000 | 0.251 | 0 |
| 8.00 | 8.662 | 1.142 | 0.031 | 1.000 | 0.251 | 0 |

Scan height is already 0.424 m at 0.605 m of travel, while uprightness is still 1.000 and knee rate is 0.259 rad/s. Knee rate next exceeds 1 rad/s after the startup window at t = 1.40 s, distance 1.366 m, speed 1.095 m/s, scan 0.717 m, gate 0. Mean absolute knee command never exceeds 1 rad after 0.5 s. Speed never falls below 0.5 m/s. The PD baseline also keeps the gate off, crosses 1 rad/s of knee rate at 1.390 m, and ends at 5.492 m by pitching. This clip is that pitch, farther along because the cruise is faster. It is not an earlier tumble. Clearance is at least the PD's. Tumble onset is not earlier than a PD that never opens the gate.

Frames: at 1 s the body is level on the blue steps. At 3 s it is pitched nose-down on the stair face, wheels on the treads. At 5 s it is level on the gray surround with the stairs behind it.

## Inverted stairs, hard

Directory: `cycle1_checkpoint_inverted-stairs_hard`.

Speed falls before the knees spin, and the cell is not exited. Final distance 6.665 m is past the PD's 2.929 m, so distance traveled is at least the PD's. The end state is not a cruise: speed 1.306 m/s, knee rate 3.050 rad/s, uprightness 0.783, scan 0.558 m, gate 0. Uprightness minimum is 0.717 at t = 2.96 s. End knee commands are about 1.77, 0.89, 2.54, 0.01 rad, not the PD's windup to about 19 rad. Mean action MSE against the PD is 2.904.

Knee-rate onset after 0.5 s is t = 1.14 s, distance 0.989 m, speed -0.175 m/s, knee rate 1.496 rad/s, uprightness 0.992, scan 0.261 m, gate still 0. Speed has already fallen below 0.5 m/s at t = 0.98 s, distance 0.943 m, while knee rate is 0.570 rad/s. The PD baseline opens the gate at 0.909 m after speed has already fallen. This onset is later than 0.909 m, and it is still after the stall. The teacher gate inside this rollout opens at 1.010 m and stays on only through 2.155 m. That is not the baseline gate, and it is also later than 0.909 m.

Frames: the 1 s grab does not show the robot, only the red stair mesh. At 2 s the body is pitched on the red stair face with the legs lifted. At 5 s and 7 s it is still on the red stairs, tilted, short of the gray surround. The cell is not cleared.

## Box grid, hard

Directory: `cycle1_checkpoint_box-grid_hard`.

This clip does clear, and the knee-rate rise starts before the PD's gate distance. End: distance 7.974 m, speed 1.141 m/s, knee rate 0.006 rad/s, uprightness 1.000, scan 0.250 m, gate 0. The PD baseline ends at 1.603 m, uprightness 0.664, gate 1. Uprightness minimum here is 0.590 at t = 1.40 s, and the body recovers. Mean action MSE against the PD is 1.473. Peak absolute knee command is 4.702 rad at t = 3.00 s, distance 2.508 m, then the commands return to the flat offset by t = 8 s. They do not wind to about 19 rad.

Knee-rate onset after 0.5 s is t = 0.56 s, distance 0.542 m, speed 1.052 m/s, knee rate 1.013 rad/s, uprightness 0.999, scan 0.422 m, gate 0. Forward scan height starts at 0.439 m, so the cubes are already in the scan. Speed does not fall below 0.5 m/s until t = 1.02 s, distance 1.017 m. The PD baseline opens the gate at 0.903 m with speed already 0.256 m/s. The student's knee-rate onset is earlier than that gate, and speed has not fallen yet. The teacher gate inside this rollout opens later, at 1.420 m, and is on for only 15 frames (through 1.740 m).

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 0.56 | 0.542 | 1.052 | 1.013 | 0.999 | 0.422 | 0 |
| 1.00 | 1.006 | 0.847 | 2.941 | 0.979 | 0.290 | 0 |
| 1.40 | 1.223 | 0.536 | 1.697 | 0.590 | 0.323 | 0 |
| 3.00 | 2.508 | 1.335 | 3.201 | 0.993 | 0.378 | 0 |
| 6.00 | 5.711 | 1.132 | 0.037 | 1.000 | 0.251 | 0 |
| 8.00 | 7.974 | 1.141 | 0.006 | 1.000 | 0.250 | 0 |

Frames: at 1 s the body is level among the green cubes. At 2 s it is pitched in the grid, wheels down, not on its side. At 5 s it is level at the edge of the grid, entering the gray surround. At 7 s it is level on the gray plane with the grid behind it.

Clearance is at least the PD's, and knee-rate onset is earlier in distance than the PD gate. The motion releases. It is not the latched windup.

## Scan ablation, pyramid stairs hard

Directory: `cycle1_checkpoint_pyramid-stairs_hard_ablate-scan`.

Zeroing `height_scan` changes the clip. The gate stays 0. Knee rate never exceeds 1 rad/s after startup (maximum after 0.5 s is 0.768 rad/s at t = 2.48 s, distance 3.866 m). Uprightness first falls below 0.98 at t = 0.92 s, distance 1.315 m, speed 1.829 m/s, knee rate 0.580 rad/s, scan 0.762 m. With the scan, that uprightness drop and the knee-rate crossing are the same frame: distance 1.366 m, knee rate 1.121 rad/s, speed 1.095 m/s. End of the ablated clip: distance 11.920 m, speed 1.469 m/s, knee rate 0.003 rad/s, uprightness 0.997, scan 0.321 m, gate 0. Uprightness minimum is 0.691 at t = 1.42 s. With the scan the clip ends at 8.662 m and 1.142 m/s.

Knee-rate onset after 0.5 s therefore changes from 1.366 m to never. Final distance changes from 8.662 m to 11.920 m. The ablation does not leave the clip unchanged. It also does not create a latched tumble in one condition and remove it in the other. Both gates stay off. The pitch distance is almost the same (1.315 m ablated, 1.366 m with the scan), and the ablated body is already going faster when it pitches. The scan changes the stair response. It does not show an earlier tumble than the pitch both clips already do.

Frames, ablated: at 1 s the body is already pitched nose-down on the blue steps. At 2 s it is still pitched, near the bottom of the stairs. At 3 s it is on the gray surround. At 7 s it is level on the gray plane. The with-scan frame at 1 s is still level.

## Hacking signs

- Knee spin on flat: no. After 1 s, knee rate is under 0.082 rad/s (medium) and under 0.052 rad/s (hard) on every frame. The startup spike dies before 0.5 s.
- High teacher match on stairs with no clearance: no. Pyramid action MSE is 0.199 and the clip ends on the flat cruise at 8.662 m. Box-grid MSE is 1.473 and the clip ends on the flat cruise at 7.974 m.
- Spinning in place: no. Flat ends near 8.98 m. Stairs and the box grid all travel several meters.
- Falling or bouncing for speed: no on flat (uprightness stays at 1.000). The box grid dips to 0.590 and recovers. Inverted stairs stay tilted and do not buy a flat cruise that way.
- Scan ablation leaves the clip unchanged: no. Knee-rate onset and final distance both move, as quoted above.

## Goal checks

Compared with the PD numbers in `BASELINE_REVIEW.md`. One seed only.

- **Flat, medium and hard: knee rate near zero, and forward tracking in the PD band. Fail.** Knee rate passes: after 1 s the means are 0.025 and 0.024 rad/s, and every frame is under 0.1 rad/s. Forward tracking fails: means are 1.132 m/s on both difficulties, against the PD mean of 0.697 m/s.
- **Hard stairs and hard box grid: clearance at least the PD's, and tumble onset earlier in distance. Fail.** Box grid passes both parts (7.974 m and upright cruise, against the PD's 1.603 m tipped; knee-rate onset at 0.542 m against the PD gate at 0.903 m). Pyramid clearance passes (8.662 m against 5.492 m) and onset fails: the gate stays off and the knee-rate crossing at 1.366 m is the same pitch the PD does at 1.390 m. Inverted distance passes (6.665 m against 2.929 m) but the cell is not cleared, and onset fails: knee rate crosses 1 rad/s at 0.989 m after speed is already -0.175 m/s, later than the PD gate at 0.909 m.
- **Scan ablation on hard pyramid stairs changes tumble onset. Pass.** Knee-rate onset after 0.5 s is 1.366 m with the scan and does not occur without it. Final distance is 8.662 m with the scan and 11.920 m without it. The pitch itself stays near 1.3 m either way. This is not a latched tumble turning on and off.
- **Two seeds agree. Fail.** Only seed 42 was trained. Two-seed agreement is not met.

The paper goal is not passed. Reward rose from 0.58 to 48.61, and that does not overturn the failed checks.
