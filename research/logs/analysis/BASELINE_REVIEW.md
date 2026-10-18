# Baseline review

**Date:** 2026-10-13
**Protocol:** `uv run python scripts/eval_cells.py` from the workspace root. Default 8 s clip, default 1 m/s forward command, seed 0, no `--ablate-scan`. One rough-test cell per clip. `medium` is difficulty 2/3. `hard` is difficulty 1. Clips and csvs are under `research/logs/analysis/baseline_<controller>_<cell>_<difficulty>/`.

**Controllers:** PD (`--controller pd`, `TeacherStudentEnv.follow_teacher`). Existing teacher checkpoint `logs/rsl_rl/walter_fs_teacher_rough-test/2026-10-09_14-37-11/model_3100.pt` (highest `model_*.pt` iteration in that folder).

**Recorded:** 10 of 20 clips. All 10 PD clips wrote `clip.mp4` and `frames.csv`. All 10 checkpoint clips failed before rollout. No training was started.

`forward_scan_height` is mean torso clearance on the forward scan rays. On flat cruise it sits at 0.267 m. A larger value is more air under those rays. `pd_tumble_gate` is the latched speed-lag flag (0 or 1). Knee commands below are the four `*_thigh_shin_joint` PD targets in radians, order torso-left, torso-right, head-left, head-right.

Frames below were taken from the mp4 with ffmpeg and looked at. The numbers are from `frames.csv`.

## Checkpoint

`model_3100.pt` does not load into the current teacher env. `scripts/eval_cells.py` builds a 16-action actor with a 383-wide observation. The checkpoint actor is 17 actions and 384 observations:

```
size mismatch for obs_normalizer._mean: copying a param with shape torch.Size([1, 384]) from checkpoint, the shape in current model is torch.Size([1, 383]).
size mismatch for distribution.std_param: copying a param with shape torch.Size([17]) from checkpoint, the shape in current model is torch.Size([16]).
size mismatch for mlp.0.weight: copying a param with shape torch.Size([512, 384]) from checkpoint, the shape in current model is torch.Size([512, 383]).
size mismatch for mlp.6.weight: copying a param with shape torch.Size([17, 128]) from checkpoint, the shape in current model is torch.Size([16, 128]).
size mismatch for mlp.6.bias: copying a param with shape torch.Size([17]) from checkpoint, the shape in current model is torch.Size([16]).
```

The same `RuntimeError` was raised for flat medium and flat hard. The extra action is the waist: the current env locks `head_joint` at 0 and removes the waist action (`cfg.actions.pop("waist", None)` in `src/walter_rl/control/full_body_pd/env.py`). This is an action-space change since that run, not a one-line bug in `scripts/eval_cells.py`, so the eval script was left as it is. The other eight checkpoint cells were not re-run; they fail in `runner.load` before the cell matters. There is no checkpoint clip to compare with the PD, so this review cannot say whether that policy only clones the teacher.

## PD, flat

Directories: `baseline_pd_flat_medium`, `baseline_pd_flat_hard`.

Both difficulties are flat ground, and the traces agree. The largest gap is 0.0035 m/s in forward speed and 0.013 rad/s in knee rate. The gate is 0 on every frame of both clips.

Medium, after the first second: forward speed stays between 0.692 and 0.700 m/s (mean 0.697). Knee rate stays under 0.017 rad/s (mean 0.0005). At t = 8.00 s, distance is 5.536 m, forward speed is 0.697 m/s, knee rate is 0.000, uprightness is 1.000, forward-scan height is 0.267 m, and the tumble gate is 0. Knee-command magnitude stays under 0.023 rad. The only knee-rate spike is at startup: 0.343 rad/s at t = 0.16 s, while speed is already 0.744 m/s, and it is gone before 1 s. Uprightness never leaves 1.000 (minimum 0.999984).

Hard ends at the same place: distance 5.536 m, speed 0.697 m/s, knee rate 0.000, scan 0.267 m, gate 0, uprightness 1.000.

Frames at 1 s, 3 s, and 5 s show the body level, all eight wheels on the gray plane, and no folded legs. At 7 s it is still level, near the edge of the patch. This is steady wheel driving at about 0.70 m/s. The 1 m/s command is not met. The shortfall is about 0.30 m/s, and the gate stays off for the whole 8 s. Knees are quiet. Nothing in the frames looks like spinning in place or bouncing for speed.

## PD, pyramid stairs

Directories: `baseline_pd_pyramid-stairs_medium`, `baseline_pd_pyramid-stairs_hard`.

The tumble gate stays 0 for all 400 frames on both difficulties. The robot pitches over the steps and is back on the flat cruise by the end.

Hard:

| t (s) | distance (m) | speed (m/s) | knee rate (rad/s) | uprightness | scan (m) | gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 0.661 | 0.697 | 0.002 | 1.000 | 0.436 | 0 |
| 1.96 | 1.390 | 1.152 | 1.066 | 0.959 | 0.737 | 0 |
| 2.50 | 1.762 | 0.512 | 0.849 | 0.924 | 0.778 | 0 |
| 6.00 | 4.099 | 0.691 | 0.089 | 1.000 | 0.267 | 0 |
| 8.00 | 5.492 | 0.697 | 0.000 | 1.000 | 0.267 | 0 |

Knee rate first exceeds 1 rad/s at t = 1.96 s, after 1.390 m, with scan height already 0.737 m (flat cruise is 0.267 m). Knee commands at that frame are about -0.51, -0.53, -0.25, -0.25 rad. They bottom near t = 3 s at about -1.51, -1.54, -0.51, -0.47 rad, then return to about 0.01 rad by t = 8 s. That is a pitch hold, then a release. It is not the running windup seen when the gate is on. Uprightness bottoms at 0.889. Final distance 5.492 m matches flat (5.536 m).

Medium is the same pattern, smaller pitch. Knee rate crosses 1 rad/s at t = 1.96 s, distance 1.384 m, scan 0.574 m, gate 0, speed 0.833 m/s. Final distance 5.481 m, speed 0.697 m/s, knee rate 0, uprightness 1, scan 0.267 m, gate 0. Scan height is already 0.380 m at t = 1.00 s while knee rate is still 0.003 rad/s.

Frames, hard: at 1 s the body is level and the blue steps are still ahead. At 3 s the body is pitched down the stair face, wheels on the treads. At 5 s it is on the lower landing, body nearly level. At 7 s it is level on the gray surround with the stairs behind it. Medium at 3 s is the same pitched-down pose on a shorter stair, and the csv shows the same recovery. The cell is cleared by driving, with the gate still off.

## PD, inverted stairs

Directories: `baseline_pd_inverted-stairs_medium`, `baseline_pd_inverted-stairs_hard`.

Here the gate does open, and the knee targets then wind without bound for the rest of the clip. Neither difficulty returns to the flat cruise.

Medium: through t = 1.00 s it is still the flat cruise (speed 0.696 m/s, knee rate 0.005 rad/s, gate 0) while scan height is already 0.200 m. Speed then collapses (t = 1.40 s, speed -0.071 m/s, distance 0.883 m, gate still 0). Knee rate crosses 1 rad/s at t = 1.56 s, distance 0.952 m, scan 0.214 m, gate still 0, speed 0.705 m/s. The gate opens at t = 1.76 s, distance 1.036 m, scan 0.217 m, knee rate 0.852 rad/s, speed 0.323 m/s, uprightness 0.984. After that the four knee targets climb together: about 0.97 rad at t = 2 s, about 9.9 rad at t = 5 s, and 18.97, 18.93, 19.13, 19.12 rad at t = 8 s. End of clip: distance 3.124 m, speed 0.195 m/s, knee rate 3.158 rad/s, uprightness 0.990, scan 0.231 m, gate 1. Uprightness stays above 0.914.

Hard opens the gate earlier and with the knees still slow. Gate on at t = 1.48 s, distance 0.909 m, scan 0.252 m, speed -0.127 m/s, knee rate 0.376 rad/s, uprightness 0.993. Knee rate crosses 1 rad/s just after that, at t = 1.60 s, distance 0.921 m, scan 0.243 m, gate 1, speed -0.218 m/s. Knee targets at t = 8 s are about 19.66, 19.65, 19.55, 19.55 rad. End: distance 2.929 m, speed 0.305 m/s, knee rate 2.942 rad/s, uprightness 0.959, scan 0.117 m, gate 1. Uprightness minimum is 0.808.

Frames, medium: at 1 s the body is level on the red plateau with the stair pit ahead. At 3 s, 5 s, and 7 s the legs are folded and the body is diagonal on the stair face. Frames, hard: at 2 s it is just starting to fold at the lip; at 6 s and 7.5 s it is still folded on the steps, short of the gray surround. Scan height at the gate is near the flat 0.267 m (0.217 m medium, 0.252 m hard), and forward speed has already fallen. The sustained spin is the post-gate windup. The cell is not cleared in 8 s.

## PD, box grid

Directories: `baseline_pd_box-grid_medium`, `baseline_pd_box-grid_hard`.

Both clips latch the gate and then wind the knees to about 19 rad. Hard falls over inside the grid. Medium stays upright and moves farther, and is still tumbling at 8 s.

Hard: scan height starts at 0.443 m, so the forward rays already see the cubes. Knee rate crosses 1 rad/s at t = 1.06 s, distance 0.721 m, scan 0.411 m, gate still 0, speed 0.826 m/s, uprightness 0.996. Knee commands there are still small (0.43, -0.27, -0.75, -0.08 rad). The gate opens at t = 1.42 s, distance 0.903 m, scan 0.364 m, speed 0.256 m/s, knee rate 0.311 rad/s, uprightness 0.982. By t = 2.40 s uprightness is 0.692, speed is -0.646 m/s, knee rate is 3.075 rad/s, distance is 1.325 m, and the gate is still 1. End: distance 1.603 m, speed -0.038 m/s, knee rate 1.930 rad/s, uprightness 0.664, scan 0.203 m, gate 1. Knee targets are about 20.5, 19.4, 18.8, 19.8 rad.

Medium: knee rate crosses 1 rad/s at t = 1.06 s, distance 0.715 m, scan 0.360 m, gate 0, speed 0.814 m/s. Gate opens at t = 1.50 s, distance 0.930 m, scan 0.308 m, speed -0.120 m/s, knee rate 1.339 rad/s, uprightness 0.953. End: distance 3.936 m, speed -0.103 m/s, knee rate 2.834 rad/s, uprightness 0.997, scan 0.316 m, gate 1. Knee targets are about 19.4, 18.6, 17.6, 18.9 rad. Uprightness minimum is 0.873.

Frames, hard: at 1 s the body is level among the green cubes. At 2.5 s, 4 s, and 7.5 s it is on its side in the grid, wheels up, and it has not reached the gray border. Frames, medium: at 1.5 s it is pitched into the cubes; at 7.5 s it is near the edge of the field, body tilted, legs folded, still in the grid. The pre-gate knee motion is a fraction of a radian. The multi-turn spin starts once the gate is on, after speed has already dropped. Neither clip is back to wheel driving.

## PD, slope

Directories: `baseline_pd_slope_medium`, `baseline_pd_slope_hard`.

The gate stays 0 on both. Knee rate never reaches 1 rad/s (hard max 0.988, medium max 0.857). The body pitches to the slope and then stands back up on the flat.

Hard: knee-command magnitude passes 0.2 rad at t = 2.02 s, distance 1.399 m, scan 0.527 m, knee rate 0.590 rad/s, speed 0.856 m/s, uprightness 0.996, gate 0. At t = 3.00 s the knee targets are -1.444, -1.298, -0.291, -0.423 rad, scan is 0.582 m, uprightness is 0.975, speed is 0.741 m/s, gate is 0. Uprightness minimum is 0.972. End: distance 5.493 m, speed 0.697 m/s, knee rate 0.001 rad/s, uprightness 1.000, scan 0.267 m, gate 0.

Medium: the same pitch starts at t = 2.06 s, distance 1.428 m, scan 0.445 m, knee rate 0.436 rad/s, gate 0. Uprightness minimum is 0.987. End: distance 5.534 m, speed 0.697 m/s, knee rate 0.001 rad/s, uprightness 1.000, scan 0.267 m, gate 0.

Frames, hard: at 2.5 s it is on the colored slope, body pitched with the surface, wheels planted. At 4 s it is still on the slope in that pose. At 7.5 s it is level on the gray flat with the slope behind it. Medium at 4 s and 7.5 s is the same, on a gentler face. No tumble. Both clips clear the slope and recover the flat cruise.

## What these baselines are

On flat, the PD is an upright wheel driver at 0.70 m/s with quiet knees. On pyramid stairs and on the slope it clears the cell by pitching, and the tumble gate never opens. On inverted stairs and on the box grid, speed falls first, the gate then latches, and the knees wind for the rest of the clip. Hard box grid also tips over. Hard pyramid stairs are cleared without that spin. The checkpoint that would show whether a trained policy only copies this pattern could not be played.
