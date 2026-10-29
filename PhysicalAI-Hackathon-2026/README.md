# Physical AI Hackathon for the Factory Floor (2026): client package

[![Website](https://img.shields.io/badge/website-hackathon.sidclab.org-ffbd00)](https://hackathon.sidclab.org)
[![Track 3](https://img.shields.io/badge/track-3-141413)](https://uwm.edu/csi/nvidia-rockwell-hackathon-fall-2026/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

An online platform for standardized submission, remote execution, and reproducible evaluation of robot policies on
physical robots.

You develop your policy your own way, for example in Isaac Sim with our starter and your own data. We then evaluate
it on our real robot, a FANUC ER-4iA: our server sends your machine an observation (`obs`), your policy returns the
next action, `act(obs)`, and this repeats until the rollout ends. Your policy runs on your machine; the cell plays
the actions you return and enforces every safety limit.

| Path | What it is |
|---|---|
| `policy.py` | **Your policy.** Edit this file. |
| `run.py` | Checks your policy on the mock, submits, and runs it on the cell. |
| `mock_cell.py` | A local stand-in for the cell, with the same API and limits (`run.py --mock` starts it). |
| `submission/submission.yaml` | Your team, model and task. |
| `submission/placement.yaml` | Where the objects go in each rollout, and its prompt (only if they change). `table_plan.png` shows the table and the axes. |
| `calibration/` | Optional: the real cameras' calibration (the starter's sim cameras already use it), for code that maps pixels or depth to robot coordinates. |
| `cell_client.py` | The client library. Do not edit it. |

A **run** is one approved submission: **10 rollouts** by default. A **rollout** is one attempt at your task, scored
success or failure; faults and interventions are noted.

## 1. Get the client package

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/SIDCLab/PhysicalAI-Hackathon-2026 && cd PhysicalAI-Hackathon-2026
uv sync
```

`uv sync` installs numpy, OpenCV and requests; add what your policy needs with `uv add` (e.g. `uv add torch`).

## 2. Wrap your sim model

Wrap the model you trained in simulation in `policy.py`: load it in `__init__`, and implement `act(obs)`, which
receives the robot's observation and returns the next action. After the action finishes, `act(obs)` is called again.
(`uv run run.py --checkpoint <path>` passes a path to `__init__`, if you prefer that to a path in `policy.py`.)

**Observation**

| | |
|---|---|
| `obs["images"]` | RGB images from the cameras you list in `CAMERAS` (table, side and/or wrist) |
| `obs["joints"]` | the robot's joint angles J1–J6 in degrees, and the gripper state (0 = open, 1 = closed) |
| `obs["tcp_pose"]` | the fingertip position and orientation `[x, y, z, w, p, r]`: position in mm, orientation in degrees |
| `obs["trial"]` | the current rollout, including the task instruction (your policy may ignore it) |
| `obs["depth"]` | only with `DEPTH = ["table"]`, `["wrist"]` or both in `policy.py`: uint16 images in mm along each camera's axis (0 = no depth), table 640 x 400, wrist 848 x 480. `calibration.geometry.depth_to_base` turns them into base-frame points (the wrist also needs `joints=obs["joints"]`). Not recorded |

**Action.** Choose one format with `ACTION` in `policy.py`:

- `ACTION = "joints"`: `[J1, J2, J3, J4, J5, J6, gripper]`. Move the robot with joint angles, in degrees (the v0.5
  twin's convention). Recommended for learned robot policies.
- `ACTION = "tcp"`: `[x, y, z, w, p, r, gripper]`. Move the fingertip to a target position and orientation; the
  robot works out the joints. Recommended for planners and AI agents.
  e.g. fingers pointing straight down: `w, p, r = -180, 0, R`.

If your machine is far from the cell, set `IMAGE_FORMAT = "jpeg"`. If `act` returns many actions at once (an
action chunk, such as ACT's), set `PREFETCH = True`: the next actions are planned while these still play, so
the arm does not stop between them. It helps when one call returns more than twice the time it takes.

## 3. Check formats and connection

```bash
uv run run.py --mock --device cpu
```

The remediation policy defaults to `../shared/yolo26x/best.pt`, 1280-pixel
inference, confidence 0.70, and CUDA device 0. Claim a GPU for a physical run:

```bash
srun --partition=teaching --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=00:45:00 --pty bash
uv run run.py --mode hover
```

Full mode uses the same simulator-independent controller as the supported Isaac
pipeline. It requests native 1920x1080 TableCam RGB plus calibrated depth, handles
missing caps before crooked caps, and supports the side-cone workflow validated in
simulation. The cone wrist handoff occurs once after missing-cap work; every crooked
cap then retains the clockwise roll. The cone remains the supplied 45-degree CAD
geometry. Its nominal gripper lean is also 45 degrees; the controller tries 45,
40, ... 0 degrees and selects the steepest pose reachable at both clearance and
contact for that perception-derived base-frame XYZ. This does not use tray/G-slot
identity. The lean may change at clearance between targets, while the cone-tip
press itself always travels vertically downward.
All qualified crooked candidates from the unobstructed batch view are retained,
so the arm occluding a farther vial after its first press does not discard that
camera-derived target. Each retained target is still locally wrist-refined before
contact; no tray/G-slot coordinate is substituted.
The controller rejects simulator-only fields; Isaac truth and contact
loads exist only in separate evaluation output. Because the real cell provides no
force measurement, shared-policy pressing is position-limited in both environments.

For supervised provisional contact, use CLI options rather than environment variables:

```bash
uv run run.py --mode full --allow-provisional-contact \
  --work-surface-z-m -0.0051 \
  --pickup-z-offset-m 0.003 --table-plane-offset-m 0.000 \
  --place-offset-m 0.010 --press-offset-m 0.010 \
  --crooked-action cone-press --cone-press-below-rim-m -0.005
```

`--crooked-action` selects `cone-press` (the side-mounted TPU cone),
`shoulder` (open-claw internal-shoulder tap),
`shoulder-close` (close 3 mm above shoulder contact, press, open before retreat),
`closed-tip` (closed fingertip-face tap), or `regrip` (camera-positioned pickup,
lift, same-vial replacement, release, and closed-tip tap). All five use the
same shared controller in simulation and physical execution.

Positive pick/place/press offsets move the TCP upward. For the cone endpoint,
positive `--cone-press-below-rim-m` moves down; negative values stop above the rim.
Start with `--press-offset-m 0.010 --cone-press-below-rim-m -0.005` for a visual
clearance trial, then reduce the upward press offset before increasing depth. The
CAD cone offset can be replaced after measurement with
`--cone-tip-from-tcp-m X Y Z` (metres).

The three base-frame XY calibration controls are independent:
`--table-cap-xy-offset-m X Y` corrects loose-cap pickup,
`--missing-cap-xy-offset-m X Y` corrects missing-vial placement/press, and
`--crooked-cap-xy-offset-m X Y` corrects the cone or other crooked-cap contact.
The physical defaults are respectively `(0.021002, -0.019611)` m,
`(-0.000503, 0.009987)` m, and `(0, 0)` m. The first two are the TableCam-calibrated
base-frame translations of the table-cap target being 20 mm too far away from the
camera and 20 mm too far right, plus the 10 mm left missing-vial miss. Crooked-cap
XY deliberately stays zero until the cone is tested.
The two normal loose-cap Z tunings live in
`config/remediation_offsets.json`: `pickup_z_offset_m` defaults to `0.003`
(the fingertip/end target is 3 mm above the local table plane) and
`table_plane_offset_m` defaults to `0.0` (signed base-Z correction). CLI flags
with the same names override the file defaults.
The older `--pick-offset-m` and `--cone-table-pick-offset-m` flags remain explicit
positive-up compatibility overrides; do not mix their sign convention into the
new settings.
Each XYZ correction is recorded in `policy_config.json` for the run.

Add `--wrist-refinement` to make TableCam/YOLO authoritative only for class,
count, and coarse target identity. The arm moves to a target-centred wrist view,
starts FP16 LocateAnything, then raises/translates/starts a safe pre-descent while
the result is computed. The VLM box centre is ray/plane backprojected with the
snapshot's measured joints and rejected if it is more than 40 mm from the YOLO
candidate. No contact waypoint is emitted until that check passes. Missing-vial
mouths and crooked caps follow the same contract; cone orientation overlaps the
first crooked-cap inference. The clockwise roll is performed only once; any
target-coordinate reach adjustment changes only the FANUC-W lean at clearance.

The cell uses one requested RGB size for both cameras. With native TableCam capture,
wrist RGB is therefore served at 1920x1080. Its calibrated 1280x720 focal length
619.63996 px scales to 929.45995 px without changing field of view. The simulator
models the physical 848x480 colour sensor before upscaling. Wrist depth remains a
separate 848x480 stream. For a loose table cap, the physical policy registers it
into wrist colour, excludes the VLM cap box, and robustly fits the surrounding
paper/table plane. Pickup Z places the fingertip at the configured clearance
above that support plane (3 mm by default). The documented 15.492 mm upright-cap
geometry is used only to convert that absolute clearance into the controller's
cap-reference convention. The fit
requires at least 24 inliers, at most 4 mm RMS, at most 5 degrees tilt, and
agreement within 15 mm of the calibrated work surface; otherwise pickup retains
the calibrated plane. Placement Z is unchanged. Missing and crooked targets do
not depend on wrist depth.

Residual hand-eye/TCP corrections remain independently tunable with
`--wrist-table-cap-xy-offset-m`, `--wrist-missing-cap-xy-offset-m`, and
`--wrist-crooked-cap-xy-offset-m`. They default to zero and do not inherit the
older TableCam empirical offsets.

On this cluster, both mock and real-cell runs can use the T4-pinned launcher:

```bash
sbatch scripts/run_physical_policy.slurm --mock --trials 1 --trial-s 8 \
  --mode full --allow-provisional-contact --crooked-action cone-press
```

The physical YOLO+FP16-VLM policy fits on the launcher's single requested T4.
Isaac plus the VLM uses two isolated T4s:

```bash
sbatch --gres=gpu:t4:2 scripts/run_compliant_breakable_pipeline.slurm \
  shared/scenes/tablecam_right_pink_spares_qualified_v2.usda \
  shared/yolo26x/best.pt 2026-10-29_wrist_vlm_full_workflow \
  --crooked-action cone-press --wrist-refinement --wrist-vlm-device 1
```

If the operator provides a direct cell token, export it as
`PHYSICAL_CELL_TOKEN` before submitting the job rather than putting the secret
in the command line. Without a direct token, `run.py` uses the private
`.request.json` created by the normal submission/approval workflow.

`--work-surface-z-m` is expressed in robot-base metres. The vial-rim plane is
derived by adding the known 10 mm tray base and 203.2 mm vial height. Use
`--vial-rim-z-m` only when directly overriding that derived rim measurement;
the policy records both effective planes in `height_planes.json`.

Start from the approved vertical home with the wrist camera facing away from the
robot base (physical-cell W/P/R `-180, 0, 0` degrees). To protect the wrist-camera wiring,
the policy refuses to start more than 20 degrees from this orientation.

Positive offsets move the TCP upward. Start at +10 mm for non-contact alignment,
then test +5 mm, +2 mm, and +1 mm one operation at a time. Zero or negative press
offsets may create contact with glass.

Runs your policy against a stand-in for the cell on your machine, with no robot. It ends with
**✓ Ready for the real robot**, or **✗ Not ready** with what to fix (an action the robot would reject, or an error
in `act`) and where it first happened. The mock's images are synthetic, so your policy may not do anything useful
there; it only checks that your policy and the cell understand each other.

## 4. Describe your task and submit

Fill in `submission/submission.yaml` (each field has a comment), and `submission/placement.yaml` unless your task
uses our default scene (8 screws and the yellow bin, the same every rollout), then send it:

```bash
uv run run.py --submit
```

Then pick a 30-minute slot in the **Calendar** tab of [hackathon.sidclab.org](https://hackathon.sidclab.org/#calendar)
with the request id it prints, such as `q3ac71f`; we approve your request for that slot. We keep adding
available slots.
In `placement.yaml`, list where each object goes and optionally a `prompt`, the instruction your policy gets in
`obs["trial"]["instruction"]`: one entry for a scene that stays the same, or one entry per rollout if objects or layout
are `different`. Send none and keep both `same`, and every rollout gets the default scene.

`--submit` also saves `.request.json`: keep that file, and keep it private (your client needs it, and it holds the
secret for your request). We review the request and approve it, or reject it with a reason. Check where it stands:

```bash
uv run run.py --status
```

## 5. Run during your slot

Once we approve your request, your client picks up its token by itself and waits for the operator. Start it and
leave it running:

```bash
uv run run.py
```

For each rollout, the operator homes the arm, sets up the objects as your submission says, and starts it; your
policy then runs until the operator scores the rollout, stops it, or the controller faults, and the next one begins.
You do nothing between rollouts. Come to the lab to watch if you can; every rollout is recorded either way.

Every rollout is recorded and appears with its score and videos under **My results** on
[hackathon.sidclab.org](https://hackathon.sidclab.org). Only your team sees them: open the link that
`uv run run.py --status` prints (`https://hackathon.sidclab.org/#key=...`); your browser remembers it.

To work with the data yourself, download your run's recorded LeRobotDataset (every recorded rollout so far; the link
**download dataset** under My results gives the same file):

```bash
uv run run.py --download
```

It goes to LeRobot's default dataset folder. Run it again after more rollouts to get the latest.

## Recording

Every rollout is recorded: the three cameras and the robot state, 10 times a second. The videos appear under My
results next to your scores, and on the leaderboard once it opens at the end of the event. Tell us when you book if
your runs should stay off it.

Local policy traces save outlined color depth visualizations under `depth/` and preserve the original uint16
millimetre frames under `depth_raw/`.

---

Organizers: UW-Milwaukee Connected Systems Institute and Milwaukee School of Engineering. Partners: NVIDIA and
Rockwell Automation. Platform developer: SIDC Lab, UW-Milwaukee.
