# Cycle 0 method

Decision: revise the plan default, then implement that revision. Do not use teacher-only reward. Do not implement residual actions in the first training run.

The plan’s default is scan-gated teacher matching plus a predictive tumble term. That default is right, and the literature does not replace it. Lee et al. 2020, Chen et al. 2020, and Miki et al. 2022 all imitate a teacher that sees more than the student. The PD teacher sees less: body state and the twist command, with tumble onset only after missed distance passes 0.2 m. Ross et al. 2011 (DAgger) and a fixed `track_teacher_action` weight both treat that action as the label to copy. Rajeswaran et al. 2018 (DAPG) is the pattern to keep instead: the task return stays in the objective, and imitation is an auxiliary. Zhang et al. 2021 (CAIL) is the reason to make the auxiliary spatial. Imitation should be strong where the demonstrator is good and weak where it is late. CAIL learns that weight from ranked trajectories we do not have. Here the split is known, so the weight is a function of the forward scan. Johannink et al. 2019 (residual RL) stays the fallback if this gate still clones the teacher. Kumar et al. 2021 (RMA) is not used: a latent code from proprioceptive history estimates contact after it happens, and the scan is already in the observation. Lee et al. 2024 supports the behavior (drive on flat, legs on a mapped obstacle) on a wheeled-legged robot, but their teacher still has the cleaner map, and their obstacle gait is a walk rather than this knee tumble.

One revision of the plan’s wording is required. “A rise in the forward half of the scan” also fires on a slope. `rough-test` slopes go up to a rise/run of 0.45 (about 0.045 m of height per 0.1 m of travel). The hard pyramid step, after the 1.5 stair scale on the 0–0.12 m range, is a 0.18 m face, and the hard box grid is 0.8 times the 0–0.3 m preset, so 0.24 m. The PD pitch loop is a good teacher on a plane and a late teacher on a face. The gate is therefore the largest upward jump along the forward height profile, not the net rise and not the height standard deviation.

## Reward changes

Implement these only on the teacher environment (`walter_fs_teacher_env_cfg`). Leave the non-teacher reward config alone. Keep PPO. The training command for the main run is `scripts/train.py --teacher --task rough-test` without `--teacher-only-reward`.

Read the gate from the `terrain_scan` ray hits, not from the `height_scan` observation. The observation is scaled by 1/5 and, on the actor, noised by ±0.1. The reward needs meters. Transform valid hits (distance ≥ 0) into the torso yaw frame. Ignore rays with x ≤ 0. Bin the forward hits by the 0.1 m longitudinal spacing, take the median height in each bin across the lateral rays, and let `step` be the maximum positive difference between consecutive occupied bins. If a bin is empty, skip it; do not treat a missing ray as a step. Do not read `terrain_levels`, `terrain_types`, or `_is_hard_stairs`. A curriculum-column gate would not move in a scan ablation.

Define the teacher match gate and the tumble gate from `step`:

- `match_gate = 1` when `step ≤ 0.04` m
- `match_gate = 0` when `step ≥ 0.08` m
- linear in between
- `tumble_gate = 1 - match_gate`

A hard slope sits near 0.045 m per sample, so it stays near full teacher match. A 0.12–0.18 m stair face and a hard box clear 0.08 m and turn the match off. These two thresholds are starting values. Change them only if a frame review shows the gate flickering on flat ground or staying shut on a hard stair. Do not retune them from the reward curve.

Replace the constant-weight `track_teacher` term with a gated term. Keep the kernel in `track_teacher_action` (exponential of mean squared action error, `std = 1`) and multiply it by `match_gate`. Keep the config weight at `TRACK_TEACHER_WEIGHT` (1.5). The per-environment scale has to live inside the reward function; the config weight is a scalar. On flat ground the peak contribution is still 1.5, under the forward tracker (weight 2.0) plus yaw (weight 1.0).

Add `predictive_knee_tumble`. Do not turn on the existing `knee_tumble`. That function deletes its `sensor_name` argument and gates on the curriculum column, so it pays for knee spin anywhere on a hard stair cell, including after the step, and it ignores the scan. The new term:

- command gate: 0 when the forward command magnitude is below 0.2 m/s (the PD `min_forward_cmd`); otherwise the sign of the forward command
- progress: `relu(body_vx * command_sign)` clamped at 1, so spinning in place does not collect the term
- knee signal: mean over the four knees of `relu(knee_velocity * command_sign)`, divided by 3.0 rad/s (`TumbleGains.rate`, the PD tumble rate)
- multiply by `tumble_gate` and by progress
- config weight 1.0

The kernel is then at most about 1, so the term stays under forward tracking. Do not multiply by the speed-lag integral. `KneeStuckTumble` already does that, and that integral is the reactive teacher.

Leave `knee_stuck_tumble` at weight 0.25. It is the recovery signal after a stall, not the predictive signal. If the first review shows tumble onset locked to the PD, set this weight to 0 on the next cycle rather than raising it.

Do not add a new progress reward. `track_lin_vel_xy` (weight 2.0) already pays for forward speed and does not penalize vertical velocity. `drive_forward` is not part of the simple teacher reward set; do not add it in cycle 0.

Inverted stairs are a drop, so `step` stays small and the predictive term stays off. The lag term still covers a stall there. The cycle-0 pass checks that name hard stairs and the hard box grid, not inverted stairs. Slopes should keep the teacher. If a review shows knee spin on a slope where the PD tracks the command upright, raise the 0.08 m threshold; do not switch the feature back to net rise.

## What does not change in cycle 0

Do not change the observation. Actor and critic already receive `height_scan`. Do not change `_TERRAIN_SCAN_SIZE` (2.4, 1.2), `_TERRAIN_SCAN_RESOLUTION` (0.1), or `_TERRAIN_SCAN_OFFSET` (0.4, 0.0). Shifting the grid forward is the first knob only after a review shows that zeroing the scan leaves tumble onset unchanged.

Do not change the `rough-test` mix. Flat cells stay so the match term has ground on which quiet knees are correct. Stair scale, row count, and time on stair columns are the first knobs only if the review says steps were too rare or too small to learn.

Do not train with `--teacher-only-reward`. That mode is the cloning baseline, scored from an existing checkpoint by the evaluation agent, not a new run.

Do not add DAgger, a second network, a privileged RL teacher, an RMA latent, or a residual action path.

## Fallback, only after a failed review

Use this only if the one-cell review of the gated run shows the student still matching the PD on hard stairs: tumble onset not earlier than the PD, and a scan ablation that does not move onset. Then, and not before:

- applied action becomes `teacher_action + policy_action` inside `TeacherStudentEnv` (the `follow_teacher` path is not this; that path replaces the student)
- remove the teacher-match reward, because the PD command is already added
- keep the task rewards and `predictive_knee_tumble`
- penalize squared residual magnitude with weight −0.1, multiplied by `match_gate`, so the penalty is on when the scan shows no step and off when it does

A residual penalty without that gate will hold the student on the reactive teacher. Residual mode is not the first training run.

## Title note for the scribe

The working title can stay: *Teacher-Student Learning of a Predictive Locomotion Policy from a Reactive Controller*. The introduction has to define the pair. In Lee et al. 2020 and Miki et al. 2022 the teacher is the better-informed policy and the student is trained to copy it. Here the teacher is the worse-informed controller and copying it is the negative result.
