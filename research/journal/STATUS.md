# Status

**Date:** 2026-10-14
**Iteration:** 5
**Hypothesis:** The cycle 3 policy tumbles and yaws the body so body-frame forward velocity still scores while the robot drives along the stairs instead of climbing them. Tracking is now world velocity along the spawn yaw, upright yaw off that heading is penalized, and the yaw command is zero. Terrain promotion uses the same heading. Training starts from a fresh initialization. The yaw-dodging weights are not resumed.

## Goal checks

| Check | Status |
| --- | --- |
| Flat hard: knee rate near zero; speed along the spawn heading near 1 m/s; yaw stays put | pass |
| Hard cells do not yaw off the stairs | pass on the four hard clips |
| Hard inverted stairs: progress along the spawn heading | fail |
| Hard box grid and hard pyramid stairs: progress along the spawn heading | pass |
| Scan ablation | not run this cycle |
| Two seeds agree | fail |

The goal is not passed. The study is unfinished. Cycle 5 clips are under `research/logs/analysis/cycle5_checkpoint_*_hard/`. Cycle 3 numbers are in `CYCLE3_REVIEW.md` and are not this run.

Flat knee rate after 1 s is near zero (means 0.016 and 0.014 rad/s). Forward speed mean is 1.027 m/s on both difficulties, about +0.027 m/s from the 1 m/s command and about +0.33 m/s from the PD mean of 0.697 m/s. Hard pyramid stairs are cleared by pitching (7.957 m against the PD's 5.492 m). Hard inverted stairs start the knee-rate rise at 0.718 m while speed is still 1.008 m/s, earlier than the PD gate at 0.909 m, and the cell is not cleared (6.820 m, still on the red stairs, uprightness 0.826). Hard box grid onset is at 0.450 m, earlier than the PD gate at 0.903 m, and the cell is not cleared (5.320 m, still in the green grid, end speed -0.186 m/s, gate 1). Cycle 2 had cleared that grid. Zeroing the scan on hard pyramid stairs moves knee-rate onset from 0.680 m to never and final distance from 7.957 m to 12.800 m. Only seed 42 was trained.

## Run

- Fresh start, no `--resume`.
- Command: `uv run python scripts/train.py --teacher --task rough-test --num-envs 2048 --max-iterations 500`
- Seed: 42
- Checkpoint: `research/logs/training/walter_fs_teacher_rough-test/2026-10-14_09-36-36/model_499.pt`

## Baselines

Unchanged. `research/logs/analysis/BASELINE_REVIEW.md`. The PD clips are the reference. Cycle 1 is `research/logs/analysis/CYCLE1_REVIEW.md`. Cycle 2 is `research/logs/analysis/CYCLE2_REVIEW.md`.
