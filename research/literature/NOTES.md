# Related work

These notes are for the paper’s related-work section. Citations match `research/paper/references.bib`. Nothing here is a WaLTER experimental result.

The information flow in this project is the reverse of the usual privileged-teacher setup. `FullBodyPdController` pitches, rolls, drives, and tumbles from body state and the velocity command. Tumbling is a hysteresis on an integrated forward speed lag (`_integrate_speed_lag`, `_tumble_gate`): the integral counts only while the forward command exceeds 0.2 m/s and the speed error exceeds 0.6 m/s, starts a tumble above 0.2 m of missed distance, and holds until the lag falls under 0.1 m. It never reads `terrain_scan`. The student already observes that scan: a 2.4 m by 1.2 m torso grid at 0.1 m, offset 0.4 m forward, so the rays run from about 0.8 m behind the torso to 1.6 m ahead (325 rays). Training currently adds an exponential match to the PD action that `TeacherStudentEnv` snapshots before the physics step (`TRACK_TEACHER_WEIGHT = 1.5`), and `KneeStuckTumble` pays for forward knee rate only in proportion to that same lag. Both terms can be maximized by cloning the reactive teacher and ignoring the scan. Cloning is the failure mode.

## Lee, Hwangbo, Wellhausen, Koltun, and Hutter, 2020

Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning quadrupedal locomotion over challenging terrain. *Science Robotics* 5(47):eabc5986, 2020. DOI 10.1126/scirobotics.abc5986. Published 21 October 2020.

**What it does.** A teacher is trained with reinforcement learning in simulation and is given privileged terrain information. A blind student, a temporal convolutional network over proprioception (joint encoders and IMU only, no camera or depth), is trained by supervised imitation of that teacher with dataset aggregation (DAgger). The deployed policy is the student. The teacher is confined to simulation. The policy also adds kinematic residuals on a trajectory-generator foot target (PMTG); those residuals are not the method’s main claim. An adaptive terrain curriculum is used for both stages. Ablations show that privileged training and proprioceptive memory matter for steps, slopes, and disturbances. The student is deployed zero-shot on ANYmal in mud, snow, rubble, vegetation, and water.

**What applies.** This is the standard citation for privileged teacher, blind student. It is also the explicit reason not to copy that recipe. Their teacher sees more than the student, and imitation is the goal because the teacher’s action is the better action. Our teacher sees less than the student, and the PD action on approach to a step is the worse action: the knees are still in the drive pose until the lag integral crosses 0.2 m. Distilling `teacher_action` with a fixed weight, or aggregating more of those labels, trains the failure mode their student is trying to escape. Their Figure 4 and the accompanying text are the right contrast in the paper.

**What does not apply.** Do not train a privileged RL teacher that sees the scan and then imitate it. That would abandon the claim, which is a predictive policy distilled from this reactive controller. Do not deploy a blind policy. Do not adopt PMTG foot phases; WaLTER’s rough-terrain mode is wheel driving plus a knee-spin tumble, not a quadruped gait. Their curriculum is not a substitute for the existing `rough-test` terrain.

## Chen, Zhou, Koltun, and Krähenbühl, 2019 / 2020

Dian Chen, Brady Zhou, Vladlen Koltun, and Philipp Krähenbühl. Learning by Cheating. Conference on Robot Learning (CoRL) 2019. *Proceedings of Machine Learning Research* 100:66–75, 2020. arXiv:1912.12294.

**What it does.** Two-stage imitation for driving in CARLA. A privileged agent sees the ground-truth layout and the other traffic participants. A vision-only student is trained to imitate that privileged agent and does not see the privileged state at deployment. Lee et al. 2020 cite this as the template for privileged learning, with one difference they state directly: their teacher is trained by RL, not by imitating expert trajectories.

**What applies.** Name it as the source of the privileged-teacher pattern. The same inversion applies: their teacher cheats by seeing more, and the student is not allowed to cheat. Our PD teacher is the one without the map.

**What does not apply.** There is no second agent to train. The PD controller is fixed. Do not add a bird’s-eye-view teacher, and do not switch the student to a camera. The student’s exteroception is the existing height scan.

## Miki, Lee, Hwangbo, Wellhausen, Koltun, and Hutter, 2022

Takahiro Miki, Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, and Marco Hutter. Learning robust perceptive locomotion for quadrupedal robots in the wild. *Science Robotics* 7(62):eabk2822, 2022. DOI 10.1126/scirobotics.abk2822. Europe PMC dates the issue 19 January 2022. arXiv:2201.08117.

**What it does.** The teacher again has privileged ground-truth terrain. The student sees proprioception plus height samples from a robot-centric elevation map and is trained by imitation. An attention-based recurrent encoder is trained to keep using proprioception when the map is missing, biased, or misleading (snow, vegetation, reflections, occlusion). On hardware, the perceptive policy lifts a foot before contact and clears higher steps than the proprioceptive baseline from Lee et al. 2020, which has to hit the step before it adapts. When the sensors are covered, the same policy falls back to that reactive behavior and still climbs, more slowly and with contacts.

**What applies.** The behavioral target. On a step that is still ahead in the height samples, the body should already be adapting; the proprioceptive policy is the late reference. Our PD controller is that late reference, implemented as a speed-lag gate rather than as a learned blind policy. Their step-clearance comparison (perceptive versus proprioceptive, same command, obstacle still in front) is the right shape for the one-cell review: tumble onset earlier in distance-to-obstacle than the PD, clearance at least as good. The scan-ablation check matches their covered-sensor trial: if covering the scan does not change onset, the policy is not perceptive.

**What does not apply.** Do not imitate a teacher that already sees perfect terrain. Our teacher cannot demonstrate the early tumble, so a DAgger loss cannot supervise it. Do not add their attention encoder in cycle 0. Do not treat the whole-grid height standard deviation as a roughness score. The comment on `_is_hard_stairs` in `rewards.py` already records that this standard deviation overlaps flat and stairs (about 0.06 m for both in that comment’s experience), which is why older reward terms gate on the curriculum column instead. A column gate would not move under a scan ablation. The cycle-0 gate has to be a forward step feature, computed from the raw ray hits.

## Rudin, Hoeller, Reist, and Hutter, 2022

Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. *Proceedings of the 5th Conference on Robot Learning*, PMLR 164:91–100, 2022. The meeting was 8–11 November 2021; the PMLR bibliographic record uses 2022. arXiv:2109.11978.

**What it does.** Massively parallel PPO on one GPU, with a game-style terrain curriculum, trains ANYmal locomotion in minutes (under four minutes on flat terrain, about twenty on uneven terrain, as reported in that paper). The paper’s own contribution is the parallel regime, the algorithm settings that regime needs, and the curriculum. It open-sourced the training stack.

**What applies.** WaLTER already trains with PPO on a curriculum terrain (`rough-test` keeps every third difficulty row, scales pyramid stairs by 1.5 from a 0–0.12 m step range, and replaces the noise column with a box grid). Rudin is the citation for that training style, not for a new reward. If stair encounters are too rare after the first evaluated run, the plan’s first terrain knob still applies: change the `rough-test` mix, and keep flat cells so quiet knees have something to learn on. Do not make that change before a video review says the mix is the problem.

**What does not apply.** Do not retune PPO, and do not import their reward. Cycle 0 does not change the optimizer. Their reported wall-clock times are not a target and are not results of this project.

## Kumar, Fu, Pathak, and Malik, 2021

Ashish Kumar, Zipeng Fu, Deepak Pathak, and Jitendra Malik. RMA: Rapid Motor Adaptation for Legged Robots. *Robotics: Science and Systems*, 2021. DOI 10.15607/RSS.2021.XVII.011. arXiv:2107.04034.

**What it does.** A base policy is trained in simulation conditioned on a latent code of privileged environment factors (friction, payload, and related parameters). An adaptation module is then trained to regress that latent from a short proprioceptive state-action history. At deployment the base policy runs fast and the adapter runs slower; there is no exteroceptive sensor. The paper’s own closing point is that a blind robot still fails on large perturbations, and that reliable walking needs onboard vision.

**What applies.** Cite it as the proprioceptive-adaptation line, and cite its own limitation: history can estimate contact properties after they affect the body, which is the same information the PD lag integral already uses. It cannot see a stair that has not been touched. The student here already has the scan. Using the scan for anticipation is the direction RMA defers.

**What does not apply.** Do not add a latent environment encoder, and do not train an adaptation module from state-action history. That would spend capacity on a reactive estimate the PD controller already computes in closed form.

## Johannink et al., 2019

Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, and Sergey Levine. Residual Reinforcement Learning for Robot Control. *IEEE International Conference on Robotics and Automation (ICRA)*, pages 6023–6029, 2019. DOI 10.1109/ICRA.2019.8794127. The page range is taken from the TUHH repository record for that DOI. arXiv:1812.03201. The preprint marks Johannink, Bahl, and Nair as equal contributors.

**What it does.** The applied command is the sum of a hand-designed feedback controller and a learned residual. The hand-designed part carries the structure it can already solve; RL only has to learn the contact residual. On a block-insertion task the residual outperforms the hand-designed controller alone and learns with fewer samples than RL from scratch. The residual is a correction, not a replacement: when the hand-designed controller is badly biased, a large residual can still compensate, but the paper’s successful residuals are small corrective motions around a controller that already aims at the right task.

**What applies.** This is the fallback, not the cycle-0 method. If a scan-gated match still clones the PD on stairs, set the applied action to `a_pd + π(obs)` and penalize the residual only when the forward scan does not show a step. On flat ground the PD is the right controller (upright, wheels driving, knees quiet), which is the regime where Johannink’s split helps. On a step the PD command is the wrong mode until the lag gate trips, so the residual would have to cancel the drive pose rather than nudge it. That is why gating comes first.

**What does not apply.** Do not start cycle 0 in residual mode. A penalty that keeps the residual small will pin the student to the reactive teacher exactly on the states where the teacher is late. Johannink’s tasks also have no exteroceptive input that makes the hand-designed controller systematically wrong before contact.

## Ross, Gordon, and Bagnell, 2011

Stéphane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. *Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics*, PMLR 15:627–635, 2011. arXiv:1011.0686. The PMLR BibTeX shortens the author names to Stephane Ross, Geoffrey Gordon, and Drew Bagnell; the paper’s title page uses the names above.

**What it does.** DAgger rolls out the learner, labels the visited states with the expert, aggregates those pairs with previous data, and retrains a stationary deterministic policy. The point of the reduction is covariate shift: a policy trained only on expert states fails on the states its own errors produce. The expert is assumed to be the policy worth imitating.

**What applies.** Use it to explain why a strong `track_teacher_action` term is already a one-step, on-policy imitation of the PD. `TeacherStudentEnv.step` copies `teacher_action` from the pre-physics state, which is the state the student acted from, and the reward is an exponential of the squared action error. Aggregating more PD labels would reduce shift toward the student’s state distribution and still teach the reactive action. DAgger does not fix a suboptimal expert. It imitates that expert more faithfully.

**What does not apply.** Do not add a DAgger data buffer, and do not train a separate supervised student. The main runs stay PPO with a gated imitation reward plus the task reward.

## Zhang, Cao, Sadigh, and Sui, 2021

Songyuan Zhang, Zhangjie Cao, Dorsa Sadigh, and Yanan Sui. Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality. *Advances in Neural Information Processing Systems* 34 (NeurIPS 2021). The proceedings abstract page does not list a page range; that range is omitted on purpose.

**What it does.** Standard imitation assumes the demonstrator is optimal. CAIL assigns a confidence to each demonstrated state-action pair, reweights the imitation loss by that confidence, and learns the confidence from an outer evaluation loss (in their implementation, a small set of trajectory rankings). The learned policy can outperform imitation of the mixed-quality data, including in the case where no demonstration is optimal. Confidence is a property of the dataset, inferred by bilevel optimization. It is not read off a sensor.

**What applies.** The training principle: downweight the teacher where it is a poor action, and keep imitating it where it is a good one. For this robot the split is known in advance. On a flat forward scan the PD is the desired drive policy. When a step is still ahead, the PD is still commanding wheels and a quiet knee pose, and that label should not be matched. The cycle-0 gate is a hand-specified confidence, not a learned one. We do not have ranked demonstrations of a better-than-PD tumble, so CAIL’s outer ranking loss has nothing to fit.

**What does not apply.** Do not implement CAIL, and do not learn a confidence network. The scan feature below is the confidence.

## Rajeswaran, Kumar, Gupta, Vezzani, Schulman, Todorov, and Levine, 2018

Aravind Rajeswaran, Vikash Kumar, Abhishek Gupta, Giulia Vezzani, John Schulman, Emanuel Todorov, and Sergey Levine. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations. *Robotics: Science and Systems*, 2018. DOI 10.15607/RSS.2018.XIV.049. arXiv:1709.10087.

**What it does.** Demonstration-Augmented Policy Gradient (DAPG) initializes from a small set of demonstrations and then optimizes the task return with an extra imitation term whose weight decays. The task reward remains. The paper reports policies that are more robust than either shaped RL from scratch or the demonstrations’ role as a pure clone; the demonstrations are a prior, not the objective.

**What applies.** This is the direct argument against `--teacher-only-reward`. That flag, in `scripts/train.py`, sets every reward weight except `track_teacher` to zero. A policy trained that way can only clone the PD, including the late tumble. DAPG’s structure is the one the main runs should keep: task rewards (forward and yaw tracking, uprightness, collision, action rate) stay on, and the teacher term is an auxiliary that is allowed to go to zero on states where the demonstrator is wrong. We do not decay the teacher term globally, because on flat ground the match should stay strong for the whole run. The decay is spatial, through the scan gate, rather than a function of iteration.

**What does not apply.** Do not pretrain by behavior cloning, and do not schedule `TRACK_TEACHER_WEIGHT` down over iterations. A global decay would also remove the flat-ground teacher that keeps the knees quiet.

## Lee, Bjelonic, Reske, Wellhausen, Miki, and Hutter, 2024

Joonho Lee, Marko Bjelonic, Alexander Reske, Lorenz Wellhausen, Takahiro Miki, and Marco Hutter. Learning robust autonomous navigation and locomotion for wheeled-legged robots. *Science Robotics* 9(89):eadi9641, 2024. DOI 10.1126/scirobotics.adi9641. The Science Robotics contents page dates issue 9(89) to April 2024. A secondary index gives 24 April 2024; that day was not on the contents page used here. The accepted-manuscript header on arXiv:2405.01792 confirms volume 9, issue 89, article eadi9641.

**What it does.** A wheeled-legged quadruped (ANYmal with driven wheels) learns a locomotion controller by privileged RL, in the Miki et al. sense: noiseless exteroception and terrain properties are available to the teacher, and the deployed policy sees IMU, joint encoders, and an onboard elevation map. The learned behavior drives on flat ground, with the leg joints nearly still, and switches to a stepping gait on stairs and large obstacles. Hybrid gait choice is not a hand-written state machine. Navigation is a separate high-level policy. The paper’s comparison point is cost of transport against a walking quadruped, which is outside this project’s claims.

**What applies.** The closest robot and the right qualitative split: wheels do the work on flat ground, legs intervene when the elevation map shows an obstacle. That is the split the one-cell review asks for (quiet knees and wheel drive on flat; knee spin before contact on stairs and the box grid). Their result is evidence that a height map is a sufficient signal for that switch on a wheeled-legged platform. It is not evidence about WaLTER, and this project has no cost-of-transport numbers to cite.

**What does not apply.** Their deployed student still imitates a teacher that saw a cleaner map than the student. Our teacher has no map. Their obstacle mode is a walk or trot, and they removed central-pattern-generator motion primitives; WaLTER’s obstacle mode is the existing knee tumble (positive knee rate about +Y, all four knees, while the wheels hold a tumble speed). Do not import their hierarchical navigation policy, their cost-of-transport reward, or their walking action space.

## Papers considered and not used as method

No further suboptimal-demonstrator algorithm is recommended. CAIL and DAPG cover the two facts the method needs: reweight a bad demonstrator, and keep the task reward so the policy can leave the demonstrator. Ranking-based reward inference (the family CAIL cites) needs a set of trajectories that are better than the teacher. We do not have those. Generating them would be a different project.

No classical model-predictive wheeled-legged controller is recommended for cycle 0. The teacher is already implemented. Replacing it with a planner that reads the scan would again give the teacher more sensing than the student and collapse the claim.
