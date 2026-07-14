# Open Duck Mini v2 Mass and Inertia Audit

## Result

The Playground model authors **2.1071407 kg** of physical mass across 15
explicit `<inertial>` elements. Its `head_assembly` is **0.406607 kg**, or
**19.2966%** of the robot total. The four-link neck/head chain is 0.5815347 kg,
or 27.5983%.

These values are source-authored, not produced from mesh volume or cooker
defaults. The model remains unchanged. A general MJCF importer defect did add a
spurious 1 kg to the empty outer `base` body; the importer now keeps a body with
no explicit `<inertial>` massless instead of leaking the SceneIR construction
placeholder into imported physics. After the fix, both imported and cooked mass
sum to 2.107140735 kg in float arithmetic, while `head_assembly` remains
0.406607002 kg.

## Sources and expansion audit

Primary source:

`/root/third_party/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml`

- SHA-256: `968b18de4e3f55b31252155f52779fa490989f5da92bc9b308e0bb4e81d6bb5c`
- The compiler resolves meshes from `assets` at line 9.
- There are no active `<include>` elements. The sensor and joint-property
  sections are already inlined at lines 25-56.
- The default blocks at lines 13-24 and 44-56 set joint, actuator, contact, and
  render attributes only. They set no body mass, inertia, geom mass, or density.
- The asset block at lines 428-455 declares 28 STL meshes. No mesh has a scale,
  and no asset or geom supplies mass/density for this model.
- The outer `base` at lines 58-62 has a free joint, a site, and the
  `trunk_assembly` child, but no direct geom and no `<inertial>`. MuJoCo 3.10
  compiles that body with zero mass.

The audit sums each direct `<body>/<inertial mass>` once. Mesh instances and
fixed child transforms do not add mass because every physical link already has
an explicit inertial record; the base wrapper contributes zero.

## Per-link mass and inertia

Every physical link uses `fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"`. All 15 tensors
are positive definite after symmetric eigendecomposition, and every principal
moment satisfies the rigid-body triangle inequality. The smallest triangle
slack is positive (`2.5349e-06 kg m^2`, on `head_pitch_to_yaw`).

| Link | Mass (kg) | Principal inertia eigenvalues, ascending (kg m²) | Source line |
|---|---:|---|---:|
| `trunk_assembly` | 0.6985260 | 0.00167606659, 0.00292718828, 0.00344489513 | 63 |
| `hip_roll_assembly` | 0.0664800 | 1.42544415e-05, 2.44553527e-05, 2.81818058e-05 | 118 |
| `left_roll_to_pitch_assembly` | 0.0751600 | 2.51170278e-05, 2.82416012e-05, 4.13873711e-05 | 136 |
| `knee_and_ankle_assembly` | 0.1240700 | 7.02745036e-05, 0.000216624550, 0.000228501846 | 151 |
| `knee_and_ankle_assembly_2` | 0.0725900 | 1.87269686e-05, 4.23935273e-05, 4.99575041e-05 | 175 |
| `foot_assembly` | 0.0752400 | 1.86964409e-05, 6.06095381e-05, 6.74949211e-05 | 195 |
| `neck_pitch_assembly` | 0.0661800 | 1.70230645e-05, 2.80043306e-05, 3.49456050e-05 | 224 |
| `head_pitch_to_yaw` | 0.0169378 | 4.28979258e-06, 8.03124437e-06, 9.78610305e-06 | 241 |
| `neck_yaw_assembly` | 0.0918099 | 3.06804376e-05, 6.94934873e-05, 7.08926750e-05 | 251 |
| `head_assembly` | 0.4066070 | 0.00107325450, 0.00168693330, 0.00245250220 | 267 |
| `hip_roll_assembly_2` | 0.0664800 | 1.42544415e-05, 2.44553527e-05, 2.81818058e-05 | 323 |
| `right_roll_to_pitch_assembly` | 0.0751600 | 2.51162814e-05, 2.82412661e-05, 4.13872524e-05 | 340 |
| `knee_and_ankle_assembly_3` | 0.1240700 | 7.09526340e-05, 0.000217579829, 0.000228781937 | 355 |
| `knee_and_ankle_assembly_4` | 0.0725900 | 1.87269686e-05, 4.23935273e-05, 4.99575041e-05 | 379 |
| `foot_assembly_2` | 0.0752400 | 1.86964409e-05, 6.06095381e-05, 6.74949211e-05 | 400 |
| **Total** | **2.1071407** | | |

The exact sum is:

```text
0.698526
+ 2 × (0.06648 + 0.07516 + 0.12407 + 0.07259 + 0.07524)
+ 0.06618 + 0.0169378 + 0.0918099 + 0.406607
= 2.1071407 kg
```

## Vendor comparison

Reference source:

`/root/third_party/Open_Duck_Mini/mini_bdx/robots/open_duck_mini_v2/robot.xml`

- SHA-256: `a134222e3b789f491b5a4a4781c6d19c91c86ad4d8481795b6b03f5eab6e0c7a`
- All common trunk, leg, foot, neck-pitch, head-pitch, and head-yaw masses match
  the Playground model.
- The vendor file has `head_assembly=0.352583 kg` at line 191 and separate left
  and right antenna-holder bodies of 0.00421629 kg each at lines 210 and 216.
  Its non-marker physical total is 2.06154928 kg.
- The Playground file instead folds antenna-holder and antenna meshes into the
  single `head_assembly` at lines 263-290 and authors that aggregate as
  0.406607 kg. The physical-total difference is 0.04559142 kg, exactly
  `0.406607 - (0.352583 + 2 × 0.00421629)`.

The vendor file is therefore an older/differently aggregated reference, not
evidence that the Playground head mass was created by import or cook.

## Importer defect and blast radius

Before this audit, `RigidBodyRecord` supplied a programmatic construction
default of 1 kg, and the MJCF parser left that value untouched when a body had no
`<inertial>`. The empty Playground base therefore imported and cooked as 1 kg,
producing a false 3.107140735 kg total.

The general fix initializes MJCF-imported bodies to zero mass and zero inertia,
then lets an explicit `<inertial>` overwrite both through the existing parser.
It contains no robot, body-name, scene, or BDX branch. Programmatic SceneIR,
URDF, USD, explicit MJCF inertials, geometry, actuators, gains, DOFs, and training
configuration are unchanged.

The behavior is covered by one reusable `LoadMjcf -> SceneIR -> CookedBlob`
pipeline regression. A single fixture verifies that:

- an empty wrapper cannot invent mass or inertia;
- an explicit inertial remains authoritative; and
- a geom-only body cannot inherit the unrelated 1 kg placeholder.

## Known MJCF gap

MuJoCo's default `compiler inertiafromgeom="auto"` derives mass, COM, and inertia
from direct geoms only when an explicit inertial is absent. It also honors geom
`mass`/`density`, default density, `inertiagrouprange`, primitive or mesh volume,
geom transforms, and multi-geom parallel-axis aggregation.

The Nuka importer does not yet implement those fields or that aggregation.
Accordingly, a geom-only MJCF body now imports honestly as massless instead of
silently receiving a fabricated 1 kg, but it still does not reproduce MuJoCo's
geom-derived inertia. Such models must author `<inertial>` until a dedicated
general inference implementation lands. This gap does not affect the Playground
mass total: all 15 physical links have explicit inertials, and its only missing
inertial is the empty base wrapper.

## Verification evidence

- RED: with the fix temporarily removed, the pipeline gate observed `mass=1`,
  `inertia=(1,1,1)` for the wrapper and geom-only body in both SceneIR and
  CookedBlob, while the explicit-inertial checks remained valid.
- GREEN: the mass-source pipeline regression passes through import and cook.
- Full host import suite: 154 tests, 149 passed, 5 skipped for an absent external
  Newton cup asset, 0 failed.
- BDX host probe: 16 imported/cooked bodies; `base=0`;
  `scene_total=2.107140735`; `cooked_total=2.107140735`;
  `head_assembly=0.406607002` in both representations.
- MuJoCo 3.10 oracle on the same Playground XML: 17 bodies including world;
  `base=0`; total body mass `2.1071407 kg`.
