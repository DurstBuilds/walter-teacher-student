# Research log

Append-only. Add a new entry under the previous one. Do not rewrite earlier entries.

## 2026-10-13 — Cycle 0 scaffold

Created the research tree and the living paper skeleton. No code under `src/` or `scripts/` was changed.

Created:

- `research/paper/main.tex` — article skeleton with the claim in the introduction and method sections. Results, numbers, and citations are placeholders.
- `research/paper/references.bib` — comment-only stub. The literature worker owns citations.
- `research/paper/figures/.gitkeep`
- `research/literature/.gitkeep`
- `research/journal/STATUS.md` — iteration 0 status.
- `research/journal/LOG.md` — this file.
- `research/logs/training/.gitkeep`
- `research/logs/analysis/.gitkeep`

Baselines have not been recorded. Training has not run. No clips, checkpoints, or measured numbers exist for this project yet.

## 2026-10-13 — Cycle 0 PD baselines

Recorded the PD and tried the existing teacher checkpoint before any new training run. No files under `src/` or `scripts/` were changed.

Checkpoint: `logs/rsl_rl/walter_fs_teacher_rough-test/2026-10-09_14-37-11/model_3100.pt` (highest iteration under `logs/rsl_rl/walter_fs_teacher_rough-test/`). It does not load. The saved actor is 17 actions and 384 observations; `scripts/eval_cells.py` builds 16 and 383, because the current env locks `head_joint` and drops the waist action. Flat medium and flat hard both died in `runner.load` with that size mismatch. The other checkpoint cells were not re-run. Zero checkpoint clips exist.

PD clips, default 8 s and 1 m/s, no scan ablation, all succeeded:

- `research/logs/analysis/baseline_pd_flat_{medium,hard}`
- `research/logs/analysis/baseline_pd_pyramid-stairs_{medium,hard}`
- `research/logs/analysis/baseline_pd_inverted-stairs_{medium,hard}`
- `research/logs/analysis/baseline_pd_box-grid_{medium,hard}`
- `research/logs/analysis/baseline_pd_slope_{medium,hard}`

Frame review is `research/logs/analysis/BASELINE_REVIEW.md`. Flat is upright wheel driving at 0.697 m/s with quiet knees and the tumble gate off (distance 5.536 m). Hard pyramid stairs are cleared by pitching; the gate stays 0 and the clip ends on the flat cruise (distance 5.492 m, speed 0.697 m/s, knee rate 0). Hard inverted stairs and hard box grid open the gate only after speed has already fallen (0.909 m and 0.903 m), then the knees wind to about 19 rad. Hard box grid ends tipped over after 1.603 m. The slope is cleared without a tumble. These clips are the reactive reference. The goal checks are not passed.

## 2026-10-13 — Cycle 1 train and eval

Reward code was left as implemented. No residual actions, no scan-size change, no terrain-mix change. Seed 42 is hardcoded in `scripts/train.py` (`cfg.seed = 42`) and in the PPO runner config. No `--seed` flag was added.

Smoke, exit 0:

`uv run python scripts/train.py --teacher --task rough-test --num-envs 8 --max-iterations 2`

That log is `research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-30-59`. It is not the evaluated checkpoint.

4096 environments ran out of memory during scene creation (`Failed to allocate 6366953472 bytes on device 'cuda:0'`) and was not used:

`uv run python scripts/train.py --teacher --task rough-test --num-envs 4096 --max-iterations 500`

Evaluated run, exit 0, CUDA, 2048 environments, 500 iterations (checkpoint `iter` 499):

`uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500`

Checkpoint: `research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-32-02/model_499.pt`.

Mean reward went from 0.58 at iteration 0 to 48.61 at iteration 499. `predictive_knee_tumble` at iteration 499 is 0.066, against `track_linear_velocity` 1.437. The curve is not the pass rule.

Eval used that checkpoint only. PD baselines were not rerecorded. Clips and the frame review are in `research/logs/analysis/CYCLE1_REVIEW.md`.

Goal checks, from that review:

- Flat, medium and hard: fail. Knee rate after 1 s is near zero (means 0.025 and 0.024 rad/s). Forward speed mean is 1.132 m/s, not the PD band of 0.697 m/s.
- Hard stairs and hard box grid: fail. Box grid clears (7.974 m, upright cruise) and knee-rate onset is at 0.542 m, earlier than the PD gate at 0.903 m. Pyramid clearance passes (8.662 m against the PD's 5.492 m) but the gate stays off and the motion is the same pitch the PD does. Inverted distance is 6.665 m against the PD's 2.929 m, the cell is not cleared, and knee-rate onset is at 0.989 m after speed is already negative, later than the PD gate at 0.909 m.
- Scan ablation on hard pyramid stairs: pass. Knee-rate onset is 1.366 m with the scan and does not occur without it. Final distance is 8.662 m with the scan and 11.920 m without it.
- Two seeds: fail. Only seed 42 ran.

The paper goal is not passed.

## 2026-10-14 — Cycle 2 resume, gate left unchanged

The cycle-2 hypothesis was that the match gate ignores a downward inverted-stair face, so the predictive term never opens and the student copies the reactive PD. The code does keep only the largest upward jump (`_max_occupied_rise` applies `relu`). The inverted-stair cell is not that face.

Cycle-1 `frames.csv` for hard inverted stairs shows `forward_scan_height` falling from 0.253 m at t = 0 to 0.215 m at t = 0.50 s (distance 0.470 m, speed 0.996 m/s), before the stall at 0.943 m. That series is clearance. A drop in the number is less air, so the terrain ahead is higher. A one-environment probe of the reward gate on the hard inverted cell, PD playback from the patch origin, reads a +0.180 m riser, match gate 0, tumble gate 1, from t = 0 through 0.80 s. The spawn is the pit floor. Hard pyramid stairs in the same probe are a downward face: step 0.000 m, match gate 1, tumble gate 0. The gate already opens on the cell that failed, and cycle 1 still tumbled late (knee-rate onset 0.989 m after speed had gone negative). The gate was not changed. Residual actions were not added. Scan size, scan offset, and the terrain mix were not changed.

No reward edit, so there was no smoke training run. Resume, exit 0, CUDA, 2048 environments, 500 additional iterations (499 through 998):

`uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500 --resume research/logs/training/walter_fs_teacher_rough-test/2026-10-13_23-32-02/model_499.pt`

Checkpoint: `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_00-33-54/model_998.pt`. Seed 42. Mean reward at iteration 998 is 53.26, against 48.61 at the cycle-1 checkpoint. The iteration-499 line in this log is 2.28 because episode lengths were re-randomized at resume; by iteration 549 the mean is 49.14.

Eval used that checkpoint only. PD baselines were not rerecorded. Clips and the frame review are in `research/logs/analysis/CYCLE2_REVIEW.md`.

Goal checks, from that review:

- Flat, medium and hard: pass. Knee rate after 1 s is near zero (means 0.044 rad/s on both). Speed means are 1.093 and 1.094 m/s. Command error is about +0.09 m/s. The gap versus the PD mean of 0.697 m/s is about +0.40 m/s and is not a failure.
- Hard inverted stairs and hard box grid: fail. Box grid clears (6.634 m, upright on the gray surround) and knee-rate onset is at 0.431 m, earlier than the PD gate at 0.903 m. Inverted onset is at 0.832 m while speed is still 1.050 m/s, earlier than the PD gate at 0.909 m, but the cell is not cleared (end 6.835 m, uprightness 0.752, still on the red stairs).
- Hard pyramid stairs: pass as a cleared pitch. End 8.345 m against the PD's 5.492 m. The gate stays off. Knee commands peak near 1.5 rad and return. The body pitches nose-down and is level on the gray surround by 5 s.
- Scan ablation on hard pyramid stairs: pass. Knee-rate onset is 1.235 m with the scan and 4.504 m without it. Final distance is 8.345 m with the scan and 10.399 m without it.
- Two seeds: fail. Only seed 42 ran.

The paper goal is not passed. The study is unfinished.

## 2026-10-14 — Cycle 3 stall credit and absolute step

Cycle 2 trained the same reward for 500 more iterations and hard inverted stairs still did not clear. The cycle-2 csv does not show speed going negative before the knees spin: knee rate crosses 1 rad/s at 0.832 m while speed is 1.050 m/s, and forward-scan height is already 0.222 m at 0.361 m of travel (flat cruise in that run ends at 0.255 m). Speed then reaches -0.001 m/s at 0.932 m with knee rate already 2.257 rad/s, and later -0.773 m/s at 1.363 m, while the robot is still on the stair. `predictive_knee_tumble` multiplied by `relu(body_vx)`, so those stall frames paid nothing.

The term now still pays for knee rate when the forward command is active and the scan gate is open, even if measured speed along that command is at or below zero. Positive speed stays clamped at 1. A flat scan keeps the gate shut. The step measure is the largest absolute jump, because the gate still ignored drops (cycle 2's hard-pyramid probe was step 0.000 m). The 0.04 m to 0.08 m ramp was not changed. An import check opened the tumble gate on both a +0.180 m rise and a -0.180 m drop, left a flat profile shut, and left a 0.045 m slope at match 0.875. Residual actions were not added: the cycle-2 inverted approach already had positive speed and an open probe gate, and action MSE there was 5.852 rather than a copy of the PD. Scan size, scan offset, and the terrain mix were not changed. No penalty was added to pull flat speed down to the PD's 0.70 m/s.

Resume, exit 0, CUDA, 2048 environments, 500 additional iterations (998 through 1497):

`uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500 --resume research/logs/training/walter_fs_teacher_rough-test/2026-10-14_00-33-54/model_998.pt`

Checkpoint: `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_01-31-29/model_1497.pt`. Seed 42. Mean reward at iteration 1497 is 51.90, against 53.26 at the cycle-2 checkpoint. The iteration-998 line in this log is 2.33 because episode lengths were re-randomized at resume; by iteration 1048 the mean is 52.12.

Eval used that checkpoint only. PD baselines were not rerecorded. Clips and the frame review are in `research/logs/analysis/CYCLE3_REVIEW.md`.

Goal checks, from that review:

- Flat, medium and hard: pass. Knee rate after 1 s is near zero (means 0.016 and 0.014 rad/s). Speed mean is 1.027 m/s on both. Command error is about +0.027 m/s. The gap versus the PD mean of 0.697 m/s is about +0.33 m/s and is not a failure.
- Hard inverted stairs and hard box grid: fail. Inverted onset is at 0.718 m while speed is still 1.008 m/s, earlier than the PD gate at 0.909 m, but the cell is not cleared (end 6.820 m, uprightness 0.826, still on the red stairs). Box onset is at 0.450 m, earlier than the PD gate at 0.903 m, but the cell is not cleared (end 5.320 m, speed -0.186 m/s, gate 1, still in the green grid). Cycle 2 had cleared the box grid.
- Hard pyramid stairs: pass as a cleared pitch. End 7.957 m against the PD's 5.492 m. The PD gate stays off. Knee commands peak at 2.019 rad and return. The body pitches nose-down and is level on the gray surround by 5 s.
- Scan ablation on hard pyramid stairs: pass. Knee-rate onset is 0.680 m with the scan and does not occur without it. Final distance is 7.957 m with the scan and 12.800 m without it.
- Two seeds: fail. Only seed 42 ran.

The paper goal is not passed. The study is unfinished.

## 2026-10-14 — Cycle 5 spawn-heading tracking

Cycle 4 trained `model_1996.pt` with a forced tumble target and was never evaluated; the run ended after the checkpoint. The behavior called out on the cycle 3 policy is a reward hack: body-frame `track_lin_vel_xy` still pays after the torso yaws along a stair face, and half of the yaw commands were nonzero, so turning off the stairs was often the task. Isotropic terrain promotion also counted a drive to the cell border.

Forward tracking is world velocity along the yaw latched at the start of the episode (`TrackLatchedForward`). `hold_spawn_heading` penalizes upright yaw away from that latch and stays silent during a pitch tumble. Yaw rate commands are zero. The terrain curriculum promotes only along the spawn yaw. Knee-stall and predictive tumble use that same along-track speed, so driving sideways does not clear the stall.

A resume from `model_1497.pt` was started and then stopped. That policy already yaws off the stairs, and continuing it is unlikely to unlearn the dodge. This cycle trains from a fresh initialization, seed 42, 2048 environments, 500 iterations.

Fresh run, exit 0, CUDA, about 46 minutes. No resume.

`uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500`

Checkpoint: `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_09-36-36/model_499.pt`. Near iteration 445 the mean reward is about 68–70 and `spawn_heading_error` is about 0.13 rad.

Hard-cell clips, 8 s, command 1 m/s forward and zero yaw. Yaw from the spawn heading never exceeds 0.5 rad on any of them.

- Flat: net travel along the spawn heading 8.59 m. Yaw stays under 0.005 rad. End knee rate 0.10 rad/s.
- Hard pyramid stairs: net along 8.42 m. Max yaw 0.38 rad. End knee rate 0.15 rad/s.
- Hard box grid: net along 8.13 m. Max yaw 0.15 rad. End knee rate 0.08 rad/s.
- Hard inverted stairs: yaw stays under 0.22 rad, so it does not turn off the stairs. Knee rate stays near the PD tumble rate (end 3.98 rad/s) and net travel along the spawn heading is −0.03 m. Path length is 8.29 m, so the body is moving without climbing.

The yaw dodge is not what these clips show. The inverted-stair failure is a tumble that does not produce progress along the spawn heading. The study is unfinished.
