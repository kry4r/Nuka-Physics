"""H1GraspEnv reward = a SUM OF NAMED TERMS (A3), mirroring the go2_rewards split.

The task: a fixed-base 2-finger gripper must CATCH a cup that descends from above
the fingertip catch plane (the A3 discriminative timing IC) and HOLD it by
force-closure friction. The reward is shaped so a TIMED, force-closure grasp scores
high while a constant-max slam (closes early -> cup deflects/falls) and a no-op
(never closes -> cup falls) both score near zero / negative.

TERMS (each a per-env (N,) cuda tensor; the env scales + sums them):
  cup_height   (+)  exp(-((cup_z - catch_z)/h_sigma)^2) GATED BY force-closure:
                    peaks when the cup sits at the catch plane AND is genuinely
                    pinched. The force-closure gate is CRUCIAL -- without it a cup
                    merely PASSING THROUGH the catch plane during its policy-
                    independent descent scores ~1 for a few steps, which lets a
                    constant-max slam (drop -> autoreset -> fresh descending cup ->
                    transit again) FARM the term across many short episodes and
                    out-score a real sustained grasp. Gating by force-closure makes
                    the positive faithful to the behaviour: only an actual grasp at
                    the plane earns it.
  force_closure(+)  BOTH fingers' normal impulse > eps (a hard AND): the genuine
                    grasp signal. Mere contact (one finger / batting) earns nothing.
                    Gated by cup_near (cup within the catch band) so a finger
                    pressing on a cup that is already falling past does not score.
  lin_vel      (-)  ||cup linear velocity||^2 GATED BY force-closure: penalize a
                    HELD cup that wobbles (anti-bat / stability), NOT a cup that is
                    simply free-falling because it was unsaveable. Gating is required
                    so an unsaveable Y-jitter free-fall earns 0 -- ungated it accrues
                    MORE under a waiting policy than a slam and inverts the gate on
                    the jittered training distribution (see the A3 report).
  ang_vel      (-)  ||cup angular velocity||^2 GATED BY force-closure: penalize a
                    held cup that tumbles (same gating rationale as lin_vel).
  action_sq    (-)  ||action||^2: effort cost -- taxes a constant-max squeeze so a
                    gentle, well-timed close is preferred (does NOT by itself make
                    const-max fail; the IC physics does -- see the A3 report).

WEIGHTS are tuned so:
  * a held cup (cup_height~1, force_closure~1, low vel, modest action) scores
    strongly positive every step;
  * a dropped cup (cup_height->0, force_closure 0) scores ~0;
  * a slammed/batted cup (high vel, dropped) scores negative.
The episodic return therefore cleanly orders good >> const-max ~ zero ~ random.

NB unlike the legged_gym Go2 reward there is NO ``only_positive_rewards`` clamp --
the velocity/effort negatives are part of the discriminative signal and must be
allowed to push a bad policy's return below a do-nothing's.
"""
from __future__ import annotations

import torch

# Reward-term weights (per control step; the horizon is ~200-300 steps).
# The velocity penalties are deliberately SMALL anti-bat NUDGES, not dominant terms.
# SIZED from the per-term breakdown (A3): a successfully held cup tumbles against the
# 2 point-fingers, so the RAW ang_vel of even a CLEAN grasp is large (~7400 over an
# episode); at the original -0.02 that term was ~-148 -- a quarter to a half of the
# positive signal -- which made a real grasp net-NEGATIVE and let a no-op out-return
# it on the jittered (training) distribution. Cut ang_vel ~33x / lin_vel ~10x so a
# held cup stays strongly positive (the anti-bat term only breaks ties between grasps)
# while const/random/zero stay ~0 (no closure -> all gated terms 0 at ANY velocity
# weight). This is the task-sanctioned "tune the weights so the gate holds".
W_CUP_HEIGHT = 1.0
W_FORCE_CLOSURE = 1.0
W_LIN_VEL = -0.005
W_ANG_VEL = -0.0006
W_ACTION_SQ = -0.01

# The cup-height gaussian width (m): a cup within ~2 cm of the catch plane scores
# near 1; one 5+ cm away scores ~0.
HEIGHT_SIGMA = 0.03
# Per-finger normal-impulse threshold (N*s) above which a finger is "in closure".
# Matches the A3 probe eps; the held gates show impulses ~8e-3, droppers exactly 0.
IMPULSE_EPS = 1e-5
# The cup is "near" the catch plane (force-closure only counts here) within this
# band of the catch plane -- excludes a cup already falling well past the fingers.
NEAR_BAND = 0.06


class H1GraspReward:
    """On-device evaluator of the grasp reward terms (all torch ops on cuda)."""

    def __init__(self, catch_z: float, device):
        self.catch_z = float(catch_z)
        self.device = device

    def compute(self, *, cup_z, cup_lin_vel, cup_ang_vel, finger_impulse, action,
                return_terms: bool = False):
        """Total reward (N,) cuda float32 = Σ weight * term.

        Inputs (all cuda float32):
          cup_z         (N,)   cup height
          cup_lin_vel   (N,3)  cup linear velocity
          cup_ang_vel   (N,3)  cup angular velocity
          finger_impulse(N,2)  per-finger normal impulse
          action        (N,2)  this step's clipped action (in [-1,1])
        With return_terms also returns a dict of the SIGNED per-term contributions.
        """
        dz = (cup_z - self.catch_z) / HEIGHT_SIGMA
        near = (cup_z > self.catch_z - NEAR_BAND) & (cup_z < self.catch_z + NEAR_BAND)
        both_closed = (finger_impulse > IMPULSE_EPS).all(dim=1)
        force_closure = (both_closed & near).to(cup_z.dtype)     # (N,)
        # cup_height is GATED by force-closure: a cup merely transiting the catch
        # plane during its (policy-independent) descent earns nothing -- only a cup
        # that is at the plane AND actually pinched scores. This is what stops a
        # constant-max slam from farming the height term via drop->autoreset->transit.
        cup_height = torch.exp(-(dz * dz)) * force_closure       # (N,)
        # The velocity penalties are GATED by force-closure too: they penalize the
        # INSTABILITY OF AN ACTUAL GRASP (a held cup that wobbles), NOT the velocity
        # of a cup that is simply free-falling because it was unsaveable (e.g. the
        # ~80% of the +-2.5 cm reset jitter that lands the cup on the un-actuated Y
        # axis). UNGATED, these terms accrue MORE on a doomed free-fall under a
        # waiting (good) policy than under a slam (const) -- inverting the gate on the
        # JITTERED training distribution PPO actually sees. Gating makes them faithful:
        # an unsaveable fall earns 0, so it cannot reward closing early.
        lin_v2 = cup_lin_vel.pow(2).sum(dim=1) * force_closure
        ang_v2 = cup_ang_vel.pow(2).sum(dim=1) * force_closure
        action_sq = action.pow(2).sum(dim=1)

        terms = {
            "cup_height": cup_height,
            "force_closure": force_closure,
            "lin_vel": lin_v2,
            "ang_vel": ang_v2,
            "action_sq": action_sq,
        }
        weights = {
            "cup_height": W_CUP_HEIGHT,
            "force_closure": W_FORCE_CLOSURE,
            "lin_vel": W_LIN_VEL,
            "ang_vel": W_ANG_VEL,
            "action_sq": W_ACTION_SQ,
        }
        total = torch.zeros_like(cup_z)
        contrib = {} if return_terms else None
        for name, term in terms.items():
            c = term * weights[name]
            total = total + c
            if return_terms:
                contrib[name] = c
        if return_terms:
            return total, contrib
        return total
