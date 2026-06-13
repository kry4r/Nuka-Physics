"""H1 union-grasp CHOREOGRAPHY -- the python-side mirror of the .nks `grasp` block.

This is the authoritative python expression of the H1 whole-body cup-grasp
choreography that the (now-deleted, M7) ``h1_union_scene_factory`` baked into
``examples/scenes/h1_cup_table.nks``: the three PD drive tables
(``drive_hold`` / ``drive_rest`` / ``drive_close``), the 12 grip (wrap-driven)
links, the close / rest-back offsets, and the approach -> close -> hold/lift ->
BITE phase schedule the dev-spike validated.

WHAT THIS MODULE IS
-------------------
Pure python DATA + logic. It

  * LOADS the choreography from the authored ``.nks`` `grasp` JSON section (no
    re-derivation -- the numbers are the factory's validated constants);
  * exposes the three drive tables keyed by link NAME, each entry carrying
    ``{target, kp, kd, tlim, grip}`` (1:1 with ``scene_metadata.hpp``
    ``GraspDriveEntry`` and the C-ABI ``H1UnionDriveEntry``);
  * provides a PHASE SCHEDULE / state machine (APPROACH -> CLOSE -> HOLD ->
    LIFT -> optional BITE) parameterised by ``close_offset`` / ``rest_back_offset``
    that selects, per phase, WHICH drive table applies, whether the grip is
    engaged, and whether the table support is present -- matching the parity
    oracle's 300-step lift choreography (``tests/scenario/h1_union_parity.cpp``)
    plus the dev-spike BITE (grip-off -> free-fall) honesty discriminator;
  * resolves the drive tables' link NAMES -> action/DOF columns given a
    SceneMap-style ``name -> dof`` mapping (the LIVE map M8's C-ABI will hand in
    from the loaded+cooked scene). It does NOT hardcode device indices.

WHAT THIS MODULE IS NOT (the M8 dependency)
-------------------------------------------
The ``.nks`` -> union nk::World C-ABI scene bindings (``nuka.Scene.load`` /
``scene.find`` / ``scene.settle`` / ``nuka.World``) are an **M8** deliverable
(plan line 497) and do NOT exist yet. So this module does not step a live world:
it carries the choreography DATA + the PD torque law + the name->DOF resolution,
structured so an M8 ``UnionWorld`` (whose ``drive_table(which)`` already exposes
0=hold/1=rest/2=close, see ``src/c_abi/union_world.cpp``) can drive it. The one
thing deferred to M8 is the actual ``name -> dof`` map (it comes from the cooked
articulation); :func:`H1GraspChoreography.resolve` takes that map as INPUT so the
choreography is M8-ready without a fake world.

The torque law mirrors the oracle's ``TableTorque`` EXACTLY::

    u = kp * (target - q[link]) - kd * qdot[link]
    if tlim > 0:  u = clamp(u, -tlim, +tlim)        # tlim == 0 -> unclamped (grip)
    if kill_grip and entry.grip:  u = 0             # the BITE

The phase timeline mirrors the parity oracle (300 steps @ dt = 1/240)::

    s <  10            drive_hold    grip on   table ON    HOLD  (settled curl held)
    10 <= s < 100      drive_rest    grip on   table ON    REST  (wrap backed off -0.25)
    100 <= s < 180     drive_close   grip on   table ON    CLOSE (close +0.18 -> grip engages)
    180 <= s < 300     drive_close   grip on   table OFF   LIFT  (friction-only hold; m*g*dt balance)
    [optional]         drive_close   grip OFF  table OFF   BITE  (kill grip -> cup free-falls)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


# The authored union-grasp scene (the .nks whose `grasp` block this module
# mirrors). Repo-relative -> absolute, so the module loads regardless of CWD.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_SCENE_NKS = os.path.join(_REPO_ROOT, "examples", "scenes", "h1_cup_table.nks")


# ---------------------------------------------------------------------------
# Drive table data (1:1 with scene_metadata.hpp GraspDriveEntry / the C-ABI
# H1UnionDriveEntry). Keyed by the imported-H1 link NAME -- cook-portable; M8
# re-resolves name -> device link + flat DOF column.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DriveEntry:
    """One reference PD drive-table row, keyed by imported-H1 link NAME.

    ``tau = kp*(target - q[link]) - kd*qdot[link]``, clamped to +/-``tlim`` when
    ``tlim > 0`` (0 -> unclamped, the grip-close rows); ``grip`` marks the 12
    wrap-driven close links (the BITE kill-switch columns).
    """

    link_name: str
    target: float
    kp: float
    kd: float
    tlim: float
    grip: bool


# The three reference drive tables. ``hold`` = the settled curl held; ``rest`` =
# the wrap backed off ``rest_back_offset`` (-0.25 rad); ``close`` = the H1.1
# close PD at ``+close_offset`` (0.18 rad). Same link order across all three; the
# 33 non-grip (leg/torso/arm) rows are identical across tables -- only the 12
# grip rows' targets shift by the offsets.
DriveTable = List[DriveEntry]


# ---------------------------------------------------------------------------
# Contact-sphere data (wrap / palm / foot), bound to imported-H1 links by name.
# Mirrors scene_metadata.hpp GraspContactSphere. Carried so a python consumer
# can introspect the grip geometry; M8's cook re-resolves link -> device link.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactSphere:
    link_name: str
    offset: Tuple[float, float, float]
    radius: float
    region: str  # "finger" | "thumb" | "palm" | "foot"
    broadphase_id: int


# ---------------------------------------------------------------------------
# Phase schedule / state machine.
# ---------------------------------------------------------------------------
class Phase(Enum):
    """The choreography phases (the approach -> close -> hold/lift -> BITE arc).

    APPROACH and REST are the pre-close phases (the wrap held / backed off while
    the cup still rests on the table support); CLOSE engages the grip; LIFT
    removes the table so the grip carries the cup by friction ALONE (the
    m*g*dt force-balance regime); BITE is the optional honesty discriminator --
    the grip is killed and the (now unsupported) cup free-falls.
    """

    APPROACH = "approach"  # drive_hold: settled curl held, table support present.
    REST = "rest"          # drive_rest: wrap backed off rest_back_offset.
    CLOSE = "close"        # drive_close: close +close_offset, grip engages.
    LIFT = "lift"          # drive_close: table removed, friction-only hold.
    BITE = "bite"          # drive_close + grip KILLED: cup free-falls (honesty).


# Which of the three .nks drive tables each phase selects.
_PHASE_TABLE = {
    Phase.APPROACH: "hold",
    Phase.REST: "rest",
    Phase.CLOSE: "close",
    Phase.LIFT: "close",
    Phase.BITE: "close",
}


@dataclass(frozen=True)
class PhaseSpec:
    """One scheduled phase: the table it drives, grip/support state, and the
    step at which it BEGINS (steps run at the choreography ``dt``)."""

    phase: Phase
    start_step: int
    table: str          # "hold" | "rest" | "close" -> selects the drive table.
    grip_engaged: bool  # False == the BITE (kill_grip in the torque law).
    table_support: bool  # False == table removed (the lift / friction-only hold).

    @property
    def drive_which(self) -> int:
        """The C-ABI ``drive_table(which)`` selector: 0=hold, 1=rest, 2=close."""
        return {"hold": 0, "rest": 1, "close": 2}[self.table]


# The default 300-step schedule, IDENTICAL to the parity oracle's lift
# choreography (tests/scenario/h1_union_parity.cpp: kWindowEnd=10, kRestEnd=100,
# kLiftAt=180, kSteps=300). BITE is NOT in the parity timeline -- it is the
# dev-spike grip-off discriminator, exposed as an OPTIONAL trailing phase a
# caller appends via :meth:`with_bite`.
_DEFAULT_WINDOW_END = 10   # APPROACH (hold) end == REST start.
_DEFAULT_REST_END = 100    # REST end == CLOSE start.
_DEFAULT_LIFT_AT = 180     # CLOSE end == LIFT start (table off here).
_DEFAULT_STEPS = 300       # total parity-oracle horizon.


@dataclass(frozen=True)
class PhaseSchedule:
    """An ordered list of :class:`PhaseSpec` with a total step horizon.

    :meth:`phase_at` resolves the active phase (and thus the drive table +
    grip/support state) for a given step, so an M8 stepper can, each step,
    pick the C-ABI ``drive_which`` and the kill-grip / table-support flags.
    """

    specs: Tuple[PhaseSpec, ...]
    total_steps: int

    def phase_at(self, step: int) -> PhaseSpec:
        """The active :class:`PhaseSpec` for ``step`` (the last spec whose
        ``start_step <= step``)."""
        active = self.specs[0]
        for s in self.specs:
            if step >= s.start_step:
                active = s
            else:
                break
        return active

    def with_bite(self, at_step: Optional[int] = None) -> "PhaseSchedule":
        """Return a copy with a trailing BITE phase (grip killed, table off) --
        the dev-spike honesty discriminator (grip-off -> the cup free-falls).

        ``at_step`` defaults to the end of the current horizon; the horizon is
        extended by the dev-spike free-fall window (120 steps, ``kFall``)."""
        start = self.total_steps if at_step is None else at_step
        bite = PhaseSpec(Phase.BITE, start, "close", grip_engaged=False, table_support=False)
        return PhaseSchedule(self.specs + (bite,), max(self.total_steps, start + 120))


# ---------------------------------------------------------------------------
# The choreography object.
# ---------------------------------------------------------------------------
@dataclass
class H1GraspChoreography:
    """The full H1 union-grasp choreography loaded from the .nks `grasp` block.

    Public surface M8's C-ABI / a stepper consume:

      * :attr:`drive_hold` / :attr:`drive_rest` / :attr:`drive_close` -- the three
        PD drive tables (lists of :class:`DriveEntry`, keyed by link name);
      * :attr:`grip_links` -- the 12 wrap-driven grip links (the close + BITE
        kill columns);
      * :attr:`close_offset` / :attr:`rest_back_offset` -- the choreography knobs;
      * :meth:`schedule` -- the default :class:`PhaseSchedule`;
      * :meth:`drive_table` -- a table by the C-ABI ``which`` index (0/1/2);
      * :meth:`resolve` -- name -> DOF column resolution against an M8-supplied
        ``name -> dof`` map (returns a :class:`ResolvedChoreography`);
      * scene scalars (:attr:`cup_mass`, :attr:`total_mass`, :attr:`gravity_z`,
        :attr:`dt`, ...) and the force-balance reference :meth:`weight_kick`.
    """

    # -- integrator / friction / cup scalars (the .nks grasp block) ----------
    gravity_z: float
    dt: float
    finger_mu: float
    table_mu: float
    foot_mu: float
    cup_mass: float
    cup_scale_xy: float
    cup_scale_z: float
    total_mass: float
    dof_stride: int
    base_dof: int
    poly_cx: float
    table_height: float
    close_offset: float
    rest_back_offset: float
    ankle_settle_tau: Tuple[float, float]

    # -- the choreography tables --------------------------------------------
    grip_links: Tuple[str, ...]
    drive_hold: DriveTable
    drive_rest: DriveTable
    drive_close: DriveTable
    dof_torque_limit: Tuple[float, ...]

    # -- the contact geometry (introspection) -------------------------------
    wrap_spheres: Tuple[ContactSphere, ...]
    foot_spheres: Tuple[ContactSphere, ...]

    # -- provenance ----------------------------------------------------------
    cup_hull_ref: str = ""
    source_path: str = ""

    # -- loaders -------------------------------------------------------------
    @classmethod
    def load(cls, nks_path: str = DEFAULT_SCENE_NKS) -> "H1GraspChoreography":
        """Load the choreography from a ``.nks`` file's `grasp` JSON section."""
        with open(nks_path, "r") as f:
            doc = json.load(f)
        if "grasp" not in doc:
            raise ValueError(f"{nks_path}: no `grasp` block (not a union-grasp scene)")
        return cls.from_grasp_block(doc["grasp"], source_path=nks_path)

    @classmethod
    def from_grasp_block(cls, g: Mapping, source_path: str = "") -> "H1GraspChoreography":
        """Build from an already-parsed `grasp` dict (the .nks `grasp` block)."""

        def _table(key: str) -> DriveTable:
            return [
                DriveEntry(
                    link_name=e["link"],
                    target=float(e["target"]),
                    kp=float(e["kp"]),
                    kd=float(e["kd"]),
                    tlim=float(e["tlim"]),
                    grip=bool(e["grip"]),
                )
                for e in g[key]
            ]

        def _spheres(key: str) -> Tuple[ContactSphere, ...]:
            return tuple(
                ContactSphere(
                    link_name=s["link"],
                    offset=(float(s["offset"][0]), float(s["offset"][1]), float(s["offset"][2])),
                    radius=float(s["radius"]),
                    region=s["region"],
                    broadphase_id=int(s["broadphase_id"]),
                )
                for s in g.get(key, [])
            )

        ast = g.get("ankle_settle_tau", [0.0, 0.0])
        return cls(
            gravity_z=float(g["gravity_z"]),
            dt=float(g["dt"]),
            finger_mu=float(g["finger_mu"]),
            table_mu=float(g["table_mu"]),
            foot_mu=float(g["foot_mu"]),
            cup_mass=float(g["cup_mass"]),
            cup_scale_xy=float(g["cup_scale_xy"]),
            cup_scale_z=float(g["cup_scale_z"]),
            total_mass=float(g["total_mass"]),
            dof_stride=int(g["dof_stride"]),
            base_dof=int(g["base_dof"]),
            poly_cx=float(g["poly_cx"]),
            table_height=float(g["table_height"]),
            close_offset=float(g["close_offset"]),
            rest_back_offset=float(g["rest_back_offset"]),
            ankle_settle_tau=(float(ast[0]), float(ast[1])),
            grip_links=tuple(g["grip_links"]),
            drive_hold=_table("drive_hold"),
            drive_rest=_table("drive_rest"),
            drive_close=_table("drive_close"),
            dof_torque_limit=tuple(float(x) for x in g.get("dof_torque_limit", [])),
            wrap_spheres=_spheres("wrap_spheres"),
            foot_spheres=_spheres("foot_spheres"),
            cup_hull_ref=str(g.get("cup_hull", "")),
            source_path=source_path,
        )

    # -- table access --------------------------------------------------------
    def drive_table(self, which: int) -> DriveTable:
        """A drive table by the C-ABI ``which`` selector: 0=hold, 1=rest, 2=close
        (matches ``src/c_abi/union_world.cpp`` ``SelectDriveTable``)."""
        return {0: self.drive_hold, 1: self.drive_rest, 2: self.drive_close}[which]

    # -- the force-balance reference ----------------------------------------
    def weight_kick(self) -> float:
        """The per-step cup weight impulse ``m * (-g) * dt`` (~8.175e-3) -- the
        dev-spike force-balance reference (Sum lambda ~ m*g*dt in the lift window)."""
        return self.cup_mass * (-self.gravity_z) * self.dt

    # -- the phase schedule --------------------------------------------------
    def schedule(
        self,
        window_end: int = _DEFAULT_WINDOW_END,
        rest_end: int = _DEFAULT_REST_END,
        lift_at: int = _DEFAULT_LIFT_AT,
        total_steps: int = _DEFAULT_STEPS,
    ) -> PhaseSchedule:
        """The default lift choreography schedule (parity-oracle timeline).

        APPROACH(hold) -> [window_end] REST(rest) -> [rest_end] CLOSE(close) ->
        [lift_at: table off] LIFT(close) -> [total_steps]. Append the BITE with
        :meth:`PhaseSchedule.with_bite`.
        """
        specs = (
            PhaseSpec(Phase.APPROACH, 0, "hold", grip_engaged=True, table_support=True),
            PhaseSpec(Phase.REST, window_end, "rest", grip_engaged=True, table_support=True),
            PhaseSpec(Phase.CLOSE, rest_end, "close", grip_engaged=True, table_support=True),
            PhaseSpec(Phase.LIFT, lift_at, "close", grip_engaged=True, table_support=False),
        )
        return PhaseSchedule(specs, total_steps)

    # -- name -> DOF resolution (the M8 seam) --------------------------------
    def resolve(self, name_to_dof: Mapping[str, int]) -> "ResolvedChoreography":
        """Resolve every drive table's + grip link's NAME to a DOF/action column
        against an M8-supplied ``name -> dof`` map (from the cooked scene).

        ``name_to_dof`` maps an imported-H1 link name (e.g. ``"h1/R_index_proximal"``)
        to its flat action/DOF column (the prefix-sum ``DofIndexOf`` order). Every
        drive-table link and grip link MUST be present; a missing name raises
        loudly (a cook/authoring mismatch). Returns the per-table column arrays +
        the grip DOF columns -- exactly what an M8 stepper needs to scatter the PD
        torques into the action buffer. Device indices are NEVER hardcoded.
        """
        def _cols(tab: DriveTable) -> List[int]:
            cols: List[int] = []
            for e in tab:
                if e.link_name not in name_to_dof:
                    raise KeyError(
                        f"drive link {e.link_name!r} absent from the scene name->dof map "
                        f"(cook/authoring mismatch)"
                    )
                cols.append(int(name_to_dof[e.link_name]))
            return cols

        missing_grip = [ln for ln in self.grip_links if ln not in name_to_dof]
        if missing_grip:
            raise KeyError(f"grip links absent from the scene name->dof map: {missing_grip}")

        return ResolvedChoreography(
            choreo=self,
            name_to_dof=dict(name_to_dof),
            hold_cols=_cols(self.drive_hold),
            rest_cols=_cols(self.drive_rest),
            close_cols=_cols(self.drive_close),
            grip_dofs=[int(name_to_dof[ln]) for ln in self.grip_links],
        )

    # -- self-consistency invariants ----------------------------------------
    def check(self) -> None:
        """Assert the choreography is internally consistent with the factory
        semantics: table sizes, grip flags, and the close/rest offset relations
        (close.target == hold.target + close_offset; rest.target == hold.target +
        rest_back_offset for the 12 grip links; non-grip rows identical across
        tables). Raises ``AssertionError`` on any mismatch.
        """
        # -- table sizes -----------------------------------------------------
        n = self.dof_stride - self.base_dof  # 51 - 6 == 45 actuated joints.
        assert len(self.drive_hold) == n, f"drive_hold size {len(self.drive_hold)} != {n}"
        assert len(self.drive_rest) == n, f"drive_rest size {len(self.drive_rest)} != {n}"
        assert len(self.drive_close) == n, f"drive_close size {len(self.drive_close)} != {n}"
        assert len(self.grip_links) == 12, f"grip_links size {len(self.grip_links)} != 12"
        assert len(self.dof_torque_limit) == self.dof_stride, (
            f"dof_torque_limit size {len(self.dof_torque_limit)} != {self.dof_stride}"
        )
        assert len(self.wrap_spheres) == 30, f"wrap_spheres {len(self.wrap_spheres)} != 30"
        assert len(self.foot_spheres) == 4, f"foot_spheres {len(self.foot_spheres)} != 4"

        # -- same link order across the three tables -------------------------
        order = [e.link_name for e in self.drive_hold]
        assert [e.link_name for e in self.drive_rest] == order, "rest link order differs"
        assert [e.link_name for e in self.drive_close] == order, "close link order differs"

        # -- grip flags match grip_links -------------------------------------
        grip_set = set(self.grip_links)
        flagged = {e.link_name for e in self.drive_close if e.grip}
        assert flagged == grip_set, "drive grip flags != grip_links"
        # base-DOF columns carry no torque limit (the 6 floating-base cols).
        assert all(x == 0.0 for x in self.dof_torque_limit[: self.base_dof]), (
            "base-DOF torque limits must be zero"
        )

        # -- the close / rest-back offset relations (the choreography law) ---
        H = {e.link_name: e for e in self.drive_hold}
        C = {e.link_name: e for e in self.drive_close}
        R = {e.link_name: e for e in self.drive_rest}
        tol = 1e-6
        for ln in self.grip_links:
            assert abs((C[ln].target - H[ln].target) - self.close_offset) < tol, (
                f"{ln}: close.target != hold.target + close_offset"
            )
            assert abs((R[ln].target - H[ln].target) - self.rest_back_offset) < tol, (
                f"{ln}: rest.target != hold.target + rest_back_offset"
            )
            # only the target shifts -- kp/kd/tlim/grip are constant across tables.
            for fld in ("kp", "kd", "tlim", "grip"):
                assert getattr(C[ln], fld) == getattr(H[ln], fld) == getattr(R[ln], fld), (
                    f"{ln}: {fld} differs across tables"
                )
        # -- non-grip rows are identical across the three tables -------------
        for e in self.drive_hold:
            if e.grip:
                continue
            ln = e.link_name
            for fld in ("target", "kp", "kd", "tlim", "grip"):
                assert getattr(H[ln], fld) == getattr(C[ln], fld) == getattr(R[ln], fld), (
                    f"non-grip {ln}: {fld} differs across tables"
                )


# ---------------------------------------------------------------------------
# A choreography resolved against a concrete scene's name->dof map (the M8 seam).
# ---------------------------------------------------------------------------
@dataclass
class ResolvedChoreography:
    """A :class:`H1GraspChoreography` with every link NAME resolved to a flat
    DOF/action column against a concrete scene map. This is what an M8 stepper
    drives: each step it picks the active phase's table + grip/support flags and
    calls :meth:`torque_columns` to get the (dof_column, tau) updates.
    """

    choreo: H1GraspChoreography
    name_to_dof: Dict[str, int]
    hold_cols: List[int]
    rest_cols: List[int]
    close_cols: List[int]
    grip_dofs: List[int]

    def _cols_for(self, table: str) -> List[int]:
        return {"hold": self.hold_cols, "rest": self.rest_cols, "close": self.close_cols}[table]

    def torque_columns(
        self,
        table: str,
        q: Sequence[float],
        qdot: Sequence[float],
        kill_grip: bool = False,
    ) -> List[Tuple[int, float]]:
        """Compute the per-DOF PD torques for one drive table, mirroring the
        oracle ``TableTorque`` EXACTLY::

            u = kp*(target - q[dof]) - kd*qdot[dof];  clamp to +-tlim if tlim>0;
            kill_grip and grip -> u = 0   (the BITE)

        ``q`` / ``qdot`` are indexed by the SAME flat DOF columns the resolution
        produced. Returns ``[(dof_column, tau), ...]`` -- the action-buffer
        scatter an M8 stepper applies (non-entry columns keep their value).
        """
        entries = getattr(self.choreo, f"drive_{table}")
        cols = self._cols_for(table)
        out: List[Tuple[int, float]] = []
        for e, dof in zip(entries, cols):
            if kill_grip and e.grip:
                out.append((dof, 0.0))
                continue
            u = e.kp * (e.target - q[dof]) - e.kd * qdot[dof]
            if e.tlim > 0.0:
                u = max(-e.tlim, min(e.tlim, u))
            out.append((dof, u))
        return out


# ---------------------------------------------------------------------------
# Self-check (runs WITHOUT the C-ABI: pure data + the .nks JSON read).
# ---------------------------------------------------------------------------
def _self_check(nks_path: str = DEFAULT_SCENE_NKS) -> H1GraspChoreography:
    choreo = H1GraspChoreography.load(nks_path)
    choreo.check()

    # table sizes (the bullet's explicit asserts).
    assert len(choreo.drive_hold) == 45
    assert len(choreo.drive_rest) == 45
    assert len(choreo.drive_close) == 45
    assert len(choreo.grip_links) == 12
    assert len(choreo.wrap_spheres) == 30
    assert len(choreo.foot_spheres) == 4
    assert len(choreo.dof_torque_limit) == 51

    # the choreography knobs match the .nks.
    assert abs(choreo.close_offset - 0.18) < 1e-6
    assert abs(choreo.rest_back_offset - (-0.25)) < 1e-9

    # resolution against a SYNTHETIC name->dof map (M8 supplies the real one).
    # Order the union links: the 6 floating-base columns, then every drive link
    # in table order, then any grip link not already seen (all are, here).
    names = []
    for e in choreo.drive_hold:
        if e.link_name not in names:
            names.append(e.link_name)
    name_to_dof = {ln: choreo.base_dof + i for i, ln in enumerate(names)}
    resolved = choreo.resolve(name_to_dof)
    assert len(resolved.hold_cols) == 45
    assert len(resolved.grip_dofs) == 12

    # the PD torque law produces one update per drive row; BITE zeros the grip.
    nq = choreo.dof_stride + 1  # cover the largest resolved column.
    q = [0.0] * nq
    qdot = [0.0] * nq
    upd = resolved.torque_columns("close", q, qdot, kill_grip=False)
    assert len(upd) == 45
    bite = resolved.torque_columns("close", q, qdot, kill_grip=True)
    grip_dof_set = set(resolved.grip_dofs)
    for dof, tau in bite:
        if dof in grip_dof_set:
            assert tau == 0.0, "BITE must zero the grip columns"
    return choreo


if __name__ == "__main__":
    c = _self_check()
    print("H1 union-grasp choreography loaded from:", c.source_path)
    print(f"  drive tables: hold/rest/close = "
          f"{len(c.drive_hold)}/{len(c.drive_rest)}/{len(c.drive_close)} entries")
    print(f"  grip links: {len(c.grip_links)}   wrap spheres: {len(c.wrap_spheres)}   "
          f"foot spheres: {len(c.foot_spheres)}")
    print(f"  dof_stride = {c.dof_stride} (base {c.base_dof} + {c.dof_stride - c.base_dof} joints)"
          f"   torque-limit cols = {len(c.dof_torque_limit)}")
    print(f"  cup_mass = {c.cup_mass}   total_mass = {c.total_mass:.4f}   "
          f"g = {c.gravity_z}   dt = {c.dt:.7f}")
    print(f"  close_offset = {c.close_offset}   rest_back_offset = {c.rest_back_offset}   "
          f"weight_kick (m*g*dt) = {c.weight_kick():.6e}")
    print()
    print("phase schedule (default lift choreography, then + BITE):")
    sched = c.schedule().with_bite()
    hdr = f"  {'phase':<9} {'start':>6} {'table':>6} {'which':>6} {'grip':>5} {'table_sup':>10}"
    print(hdr)
    for s in sched.specs:
        print(f"  {s.phase.value:<9} {s.start_step:>6} {s.table:>6} {s.drive_which:>6} "
              f"{str(s.grip_engaged):>5} {str(s.table_support):>10}")
    print(f"  total horizon: {sched.total_steps} steps @ dt={c.dt:.7f}")
    print()
    # a couple of phase_at probes (the step -> table/grip selection an M8 stepper does).
    for step in (0, 5, 50, 120, 200, sched.total_steps - 1):
        ps = sched.phase_at(step)
        print(f"  step {step:>4} -> {ps.phase.value:<9} (table={ps.table}, "
              f"grip={ps.grip_engaged}, table_support={ps.table_support})")
    print()
    print("SELF-CHECK PASSED")
