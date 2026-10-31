# TableCam ↔ Real Cell Misalignment: Hypotheses, Evidence and Tests

Date: 2026-10-28. Scope: the physical-cell runs of 2026-10-26 and 2026-10-28
(`shared/runs/*physical*`). Assumption, per the team: **`cameras.json` is the
source of truth for intrinsics and distortion** (fx, k1, fixed principal point).
Other teams used it successfully, so nothing below changes the lens model.

**Symptom**, in the team's camera-view convention (§0):
- vials are missed **too far left**, which is robot **−y**;
- spare table caps are missed **too far right**, which is robot **+y**;
- both are also missed in height.

A single constant offset cannot produce opposite-sign misses, so at least one
position-dependent or pose-dependent error must be present.

Everything here is reproducible with the scripts in
[`scripts/tools/camera_alignment/`](scripts/tools/camera_alignment/). Their outputs
go to `shared/runs/YYYY-MM-DD_camera_alignment/<test>/`, and today's evidence is
in `shared/runs/2026-10-28_camera_alignment/`.

---

## 0. Direction conventions

The TableCam sits **in front of the tray, facing the robot**. Its eye is at
base (0.776, −0.043, 0.801) m, looking toward −x and 52.6° below horizontal.

The team describes misses **from the camera's point of view**. The team's axes
are written **x_cam / y_cam** in this report so they aren't confused with the
robot's x/y, which the code, traces and scripts use:

| Team axis (camera view) | Meaning | Robot base frame |
|---|---|---|
| x_cam | right / left | **+y / −y** |
| y_cam | toward / away from the camera | **+x / −x** (away = toward the robot) |
| height | up / down | +z / −z |

So "vials too far left" = robot −y and "caps too far right" = robot +y; the
opposite-sign misses are along **robot y**. Script outputs are in robot base
coordinates; convert with the table above.

**Perspective trap.** Vertical lines in this camera converge to a point below
the image, at pixel **(906, 1322)** (see `t1_parallax_map.py`). Something that is
too high therefore appears to lean *away from* that point:

- left of column u ≈ 906 (the left tray columns), it reads as "too far left" (x_cam);
- right of it (spare caps, u ≈ 1300–1450), it reads as "too far right" (x_cam).

**A pure height error can look like "vials left, caps right" in the TableCam
image** even when the XY is perfect. Check the misses from the side camera, or
measure them with a ruler, before treating them as XY errors.

---

## 1. What the real-cell path actually does

`PhysicalAI-Hackathon-2026/policy.py` → `RemediationVisionDetector.detect`
(`src/inspection/vision_pipeline.py`) → `RemediationController`
(`src/remediation/controller.py`):

1. **XY (and the Z of most targets)**: the YOLO **box-centre pixel** is
   back-projected with `cameras.json` (`pixel_to_base`) and intersected with an
   **assumed horizontal plane** for each class:
   - `table_cap`: work surface −5.1 mm + plug 13.46 mm + flange 2.03 mm = **10.4 mm**
   - `missing_cap`: vial rim = −5.1 + 10 (tray base) + 203.2 = **208.1 mm**
   - `crooked_cap` / `good_cap`: rim + flange = **210.1 mm**

   Today's runs overrode the rim to 222.9 mm (`15-31-48`) and 203.9 mm
   (`15-36-13` onward).
2. **Depth** (640×400 `table_depth`) is attached afterwards and does not move
   XY. It only supplies `surface_top_base_z_m` and the cap axis, which set the
   **crooked-cap press height**.
3. **Commands** are UTOOL1 (the fingertip, 293 mm from the J6 origin per
   `FRAMES.md`) in UF0 (base − 330 mm).
   - Missing caps: picked and placed with W/P/R (−180, 0, 0) plus
     `--pick/--place/--press-offset-m`.
   - Crooked caps with `cone-press`: pressed at a **tilted wrist**, W/P/R
     (−180, 45, −90) in run `15-40-24`, using the CAD cone offset
     `cone_tip_from_tcp_m = (0, −36.1, −10.8) mm`.
4. `refine_defect_centres_from_grid` (the tray-lattice snap) is **not called**
   on the physical path. Each target comes from a single box centre.

**Coupling to know about.** Changing `--vial-rim-z-m` or `--work-surface-z-m`
moves the target in **XY as well as Z**: each 10 mm of plane moves it 6–10 mm
along the camera ray (table in §3, H1). Using the rim override as a "height
fudge" therefore also moves the tray targets. Pure height corrections belong in
the pick/place/press offsets, which only change Z.

---

## 2. Evidence gathered today (offline, from existing traces)

| # | Check | Result | Points to |
|---|---|---|---|
| E1 | Commanded TCP projected into the real frames (`evidence/commanded_tcp_vs_real_gripper.jpg`, `t3_tcp_overlay/`) | At the spare-cap pick (`15-22-05/0005`, `15-40-24/0005`) the projected TCP sits **on** the detected cap. The drawn tool axis runs up the middle of the real gripper. The real fingertips end visibly **above and toward +y** of the projected tip, roughly 15–25 mm up the axis (read by eye; not a measurement). | Camera, robot base and tool direction agree; the **fingertip (UTOOL1) position** is off (H2) |
| E2 | FK of the home joints + 293 mm vs the reported home TCP | FK gives UF0 (300.0, 0.0, −90.1) mm, which is exactly what the real cell reports at home | The controller still uses the **stock 223 mm-past-flange UTOOL1** |
| E3 | `src/remediation/config/custom_finger_contract.json` | `tablecam_tcp_checked: false`; every `required_measurements_m` (pick, place, press offsets) is `null` | The custom-finger TCP has **never been measured** (H2) |
| E4 | Wrist-camera origin from `cameras.json` projected into the table image | It lands at the flange, but the silver box that appears to be the wrist camera hangs beside the fingers | Tooling was **rebuilt after the 10-07 calibration** |
| E5 | T5: static scene registration, `10-26_15-41-05/0000` vs `10-28_15-22-05/0000` | 0.0 px shift, −0.001°, scale 0.9993 (482 inliers) | TableCam **did not move between 10-26 and 10-28**; drift since 10-07 is untested (H3) |
| E6 | T2: tray lattice pitch vs assumed plane (28 tray detections, `15-22-05/0001`), against the twin's 51.3 × 61.2 mm pitch | Matching the y pitch needs the plane at **213.5 mm**; matching the x pitch needs **227.0 mm**. Assumed: 210.1 mm | Plane probably **too low** (H1), but the x/y disagreement (13.5 mm) is not a pure height error: either the real pitch ≠ twin (unverified, see `docs/PHYSICAL_MEASUREMENT_CHECKLIST.md`) or the camera tilt is off (H3) |
| E7 | T6: depth heights in the tray boxes | Good-cap tops at **201.8 mm** (p10–p90 199–205), plane 210.1. Spare caps 1–5 mm, plane 10.4. Work surface −4.8 mm. Registration near the tray: median 0 px, p90 6 px | Depth says the plane is **too high**, contradicting E6. Stereo depth at ~0.75 m is roughly 1–2% (≈ ±7–15 mm), so neither depth nor the twin pitch settles the height; **a ruler does** (H1, H6) |
| E8 | Predicted outlines drawn on the real image (`evidence/predicted_outlines_*.jpg`) | Missing-vial rim and spare-cap flange outlines land on the objects within a few px at the inspection pose | Detection-centre bias is small at the inspection pose (H5 is second-order) |
| E9 | Today's configs | `pick_offset_m` −0.022, then −0.002; rim overridden to 222.9 / 203.9 mm | The team has been tuning height and plane by hand; per the coupling in §1, the rim changes also moved XY |
| E10 | Team report (2026-10-28) | Misses have a toward/away (robot x) part, but it is **smaller** than the left/right (robot y) part | Rules out a plane/height error (H1) as the main cause; it predicts the opposite ratio. Favours H2 |
| E11 | Team report (2026-10-28) | The smaller toward/away part is **toward the robot** (robot −x) for **both** vials and caps | Exactly what a plane set **too low** (H1) predicts for every target, and consistent with E6 (not E7). Each mm toward the robot implies the plane is ~1.3–1.7 mm too low. Under H1 the commanded Z is also too **low**, while a short fingertip (H2) leaves the tip too **high**: the sign of the height miss separates them |

---

## 3. Hypotheses (ranked)

Each hypothesis lists the mechanism, why it can give opposite-sign misses,
today's evidence, and the test that settles it.

**Ranking, using the team's observation (E10).** The misses are mostly
left/right (x_cam = robot y) with opposite signs for vials and caps, plus a
*smaller* toward/away part. Ask of each hypothesis whether it can make
left/right the larger component *and* flip its sign:

| Hypothesis | Left/right vs toward/away it predicts | Flips left/right sign between vials and caps? | Fit |
|---|---|---|---|
| H2 tool offset (pose-dependent) | Cone pose: a tool-length error is pure left/right + height. Vertical pose: a finger-spread offset is pure left/right | yes, if vials and caps use different wrist poses | **leading** |
| H5 per-class detection bias | any; it is class-specific | yes | possible second |
| H7 grasp transfer | mostly toward/away (jaws re-centre left/right) | only if the pockets don't centre | weak |
| H1 plane height / H3 camera distance | toward/away is **larger** (vials ≈ 6–8 mm toward the robot vs ≈ 0–1 mm left/right per 10 mm) | yes, but small | **not the main left/right cause**, but the best fit for the smaller toward-robot part on both vials and caps (E11); adds a little 'right' to the caps |
| H3 camera roll | toward/away, opposite on the two image sides | no (wrong axis) | unlikely |
| H4 robot yaw about base z | perpendicular to the radius; same left/right sign at the tray and the caps | no | unlikely |

Working model: **H2 (left/right) + H1 (toward the robot)**, which are
independent and additive. Two answers confirm or break it:
- Were the vial misses **cone-press** runs?
- Were the height misses **too high** (fingertips stop short, H2) or **too
  low** (plane too low, H1)? Or a mix: too high on vertical picks, and shifted
  in the cone pose?

On the first question: were the vial misses **cone-press**
runs? If they were vertical (`shoulder`) runs, H2 needs a finger-spread offset
that differs between the vial and cap actions, and H5 moves up.

The hypotheses keep their original numbering below; H1 is still described first
because it drives the height misses and is the cheapest to settle with a ruler.

### H1: Height planes are wrong, so XY is wrong through parallax
- **Mechanism.** Target = box-centre ray ∩ assumed plane. If the real feature is
  `dz` above the plane, the computed point lands `dz` too low **and** farther
  from the camera along the ray. For a plane 10 mm too low:

  | Target (pixel) | Robot (x, y) error | Team view |
  |---|---|---|
  | tray left (821, 663) | (−5.8, −1.2) mm | 5.8 away, 1.2 left |
  | tray centre / missing (914, 538) | (−7.7, −0.3) mm | 7.7 away, 0.3 left |
  | tray right column (1181, 669) | (−5.9, +2.8) mm | 5.9 away, 2.8 right |
  | spare cap (1305, 631) | (−6.5, +4.3) mm | 6.5 away, **4.3 right** |

  The left/right (x_cam) error flips sign across the camera's own column (robot
  y = −43 mm): **left-side vials go left, spare caps go right**, which matches the
  symptom's sign.

  But H1 makes the **away-from-camera (y_cam) error the largest one**, at
  6–8 mm per 10 mm of plane error, while x_cam stays small for vials near the
  centre. If H1 were the main cause, misses along y_cam (toward/away) would be
  bigger than the left/right misses. If the team mostly sees left/right misses,
  H1 is secondary and H2 is the better candidate. The image looks even more
  left/right than the robot error does (§0).
- **Class-specific variants.**
  - Spare caps sit on a **pink sheet of unknown thickness**, which isn't
    modelled (the plane assumes bare table). They may also not rest on their
    plug the way the 10.4 mm plane assumes.
  - The vial rim is derived from specs (10 + 203.2 mm) rather than measured.
- **Evidence.** E6 (plane low by 3–17 mm if the twin pitch is real), E7 (high by
  about 8 mm per depth), E9.
- **Tests.** Ruler (workflow step 1), then T2 and T1; finally T8 model L1.

### H2: UTOOL1 is not at the custom fingertips (a tool-frame offset)
- **Mechanism.** The controller still uses the stock 293 mm tool (E2). The custom
  fingers, their pocket ledge (2.7 mm above the tip in CAD) and the remounted
  hardware (E4) were never measured (E3). An offset fixed in the tool frame maps
  to the robot frame differently in the two wrist poses the policy uses:

  | Tool-frame axis | Vertical pick/place (−180, 0, 0) | Cone press (−180, 45, −90) |
  |---|---|---|
  | tool z (along the fingers) | robot (0, 0, −1): **height only** | robot (0, **+0.71**, −0.71): **left/right + height** |
  | tool y (finger spread) | robot (0, −1, 0): left/right | robot (−1, 0, 0): toward/away |
  | tool x | robot (+1, 0, 0): toward/away | robot (0, −0.71, −0.71): left/right + height |

  - **Worked example.** Suppose the real fingertips stop 20 mm short of UTOOL1
    (tip above the TCP, which E1 suggests).
    - Vertical cap pick: the tip is just **20 mm high**.
    - Cone-pressed vials: the tip lands at 20 × (0, −0.71, +0.71) = robot
      (0, −14, +14) mm, i.e. **14 mm left and 14 mm high**.
    - One length error therefore gives "vials left + height" and "caps height".
      A small tool-y offset (finger spread off the J6 axis) then explains "caps
      right": E1 shows the fingertips toward +y (right) of the commanded tip at
      the cap pick.
  - This needs the vial misses to be **cone-press** actions: run `15-40-24` used
    `cone-press`; the other runs today used the vertical `shoulder` tap. Check
    which runs showed "vials left". If the vertical shoulder runs show it too,
    the cone explanation doesn't apply to them.
  - The CAD `cone_tip_from_tcp_m` (0, −36.1, −10.8) mm is itself unmeasured and
    adds directly to the cone-pose error.
- **Evidence.** E1, E2, E3, E4, E9 (a −22 mm pick offset was tried).
- **Tests.** T7 hover at R = 0 and −90 (and the cone pose if possible), T3
  triangulation with the side camera, T8 tool models T0/T1/T2.

### H7: Grasp transfer turns a cap-pick miss into an opposite vial-place miss (mainly toward/away)
- **Mechanism.** Placement error = vial localization error − cap localization
  error, along any axis where the closing jaws don't re-centre the cap. A pick
  that lands +e off the cap places the cap −e off the vial.
  - At the vertical pose, the jaws close along tool y = robot ∓y, which is
    **left/right (x_cam)**. A parallel gripper re-centres the cap along that axis,
    so a left/right pick error is *not* mirrored at placement.
  - The mirror happens along the **toward/away (y_cam)** axis. H7 can therefore
    explain opposite-sign y_cam misses, but not the reported left/right ones,
    unless the pockets fail to centre (cap caught on one pad).
  - The custom pockets may not centre like flat jaws, so this is still worth the
    one supervised check.
- **Evidence.** None yet; the traces never image the held cap.
- **Test.** T7 `--grasp-check` (supervised contact): image the held cap 60 mm
  above the table from the table and side cameras.

### H3: The TableCam extrinsics are not current
- **Mechanism.** The table camera was solved on 2026-10-07, before the testbed
  rebuild, and never re-solved (the side camera was). A small rotation produces
  position-dependent errors:
  - a roll about the optical axis moves the image-left and image-right sides in
    opposite **toward/away (y_cam)** directions;
  - an error in the camera's distance along its axis (or its height) scales the
    scene about the image centre, so left targets move left and right targets
    move right. That is the reported **x_cam** pattern, and it is
    indistinguishable from a common H1 plane error until two heights are
    compared (T7 hovers at 40 vs 80 mm, or T2 + ruler).
- **Evidence.** E5 shows no motion from 10-26 to 10-28. The sim edge overlay and
  the tool-axis projection (E1) rule out large errors. The x/y pitch
  disagreement (E6) leaves room for a small tilt error.
- **Tests.**
  - T5 against any frame from on or near 10-07 (the recorded LeRobot dataset
    via `run.py --download` has the earliest frames).
  - T8 model L2. If it is supported, re-solve the camera pose with ArUco; keep
    the lens model.

### H4: The robot frame differs from the camera's base frame
- **Mechanism.** The ArUco solve defines "base". If UF0 is not base − 330 mm, if
  the base was re-mastered or moved during the rebuild, or if a J1 zero is off,
  every command is transformed. A yaw about base z gives errors perpendicular to
  the radius from the robot, in different directions for the tray (azimuth
  ≈ −6°…+20°) and the spare caps (≈ +45°).
- **Evidence.** E2 shows the FK and the controller agree at home. That checks the
  software convention, not the physical registration. E1 (tool axis on the real
  gripper) argues against large errors.
- **Tests.** T4 on probe traces (FK vs reported TCP across many poses), then
  T3/T8 residuals that grow with distance from the base.

### H5: Detection-centre bias per class
- **Mechanism.** The box centre isn't the feature:
  - a crooked cap's box follows the leaning flange;
  - a spare cap's box spans its top and side, so the centre is below the top;
  - the detector was trained in sim, so real box tightness can differ.

  Each of these is an effective per-class plane error, and acts like H1.
- **Evidence.** E8: small (a few px) at the inspection pose.
- **Tests.** T3 clicks on the true feature centre, then T8 per-class residuals.

### H6: Depth height and registration (crooked-cap Z only)
- **Mechanism.** The crooked-cap press height comes from the 90th percentile of
  the depth points in the box. `table_depth` was "tilted and shifted" only to
  flatten the table (rms 2.3 mm). Errors in object height and stereo noise at
  ~0.75 m go straight into the press Z.
- **Evidence.** E7: cap tops read about 8 mm below the plane; spare caps read
  only the sheet underneath them.
- **Tests.** T6 after the ruler step; if it is off, re-solve the depth-to-colour
  extrinsic, or press from the plane height plus a measured offset instead of
  depth.

### Ruled out or deprioritised
- **Sim camera mismatch.** The sim TableCam matches `cameras.json` within 1.5 mm
  and 0.1°. It is not a cause of real-cell misses.
- **Lens model.** Trusted per the team assumption.
- **Camera bumped between 10-26 and 10-28.** E5.
- **Image scaling or cropping.** Frames are native 1920×1080, and
  `pixel_to_base` rejects aspect changes.

---

## 4. Test scripts

All of these run with `.conda_env/bin/python`. Offline tests read
`shared/runs/<run>/` traces; T7 needs an approved cell slot.

| Test | Script | Hypotheses | Needs | Decides |
|---|---|---|---|---|
| — | ruler + calipers (no script) | H1, H5 | someone at the cell | rim height above the table; height of the spare-cap top (on the sheet) above the table; tray pitch in x and y; sheet thickness |
| T1 | `t1_parallax_map.py <run> <frame> --dz-mm 10` | H1 | any trace | how much and which way each target moves per mm of plane error; draws the vanishing column |
| T2 | `t2_tray_pitch_height.py <run> <frame> --pitch-x-mm .. --pitch-y-mm ..` | H1, H3 | ruler pitch | the plane height that reproduces the real tray. If x and y disagree by > 5 mm, H3 or a tray mismatch |
| T3 | `t3_tcp_overlay.py <run> [--clicks clicks.csv]` | H2, H4, H5 | trace (+ side image for triangulation) | where the robot thinks the fingertip is vs. where the cameras see it |
| T4 | `t4_fk_vs_tcp.py <probe_run>` | H2, H4 | probe trace (joints + tcp_pose) | whether UF0 and UTOOL1 are what the code assumes, across wrist rolls |
| T5 | `t5_camera_drift.py <ref.png> <cur.png> [--mask ..]` | H3 | two frames | whether the camera moved; px and mm |
| T6 | `t6_depth_heights.py <run> <frame>` | H6, H1 | `depth_raw/` | depth heights per class; depth→RGB registration |
| T7 | `t7_probe_policy.py [--mock] --hover-mm 40 80 --r-deg 0 -90 [--grasp-check]` | all | cell slot | non-contact hovers over left, centre and right tray targets and the spare caps, recording the table, side and wrist cameras plus joints; optional supervised grasp check (H7) |
| — | `click_points.py <probe_run> --label tip` | — | display | writes the T3 click CSV (table + side pixels) |
| T8 | `t8_fit_error_model.py measurements.csv` | all | measured pairs | which model explains the misses (L0/L1/L2 localization, T0/T1/T2 tool), with the correction values |

**How to read T7 hovers** (the cheapest discriminator):
- The tip-to-target offset is the **same at 40 and 80 mm**, but changes with R →
  tool-frame offset (H2).
- The offset is the same at both heights and both rolls → base-frame offset (H4).
- The offset **changes with hover height** along the camera ray → localization
  (H1/H3/H5). The fingertip is on the commanded line; the target isn't.
- The offset has **opposite left/right (x_cam, robot y) sign** on the left of
  the tray vs the spare caps at both heights → H1 (plane) or H3 (camera
  distance).
- Left/right appears only in the cone pose → H2 tool length (or cone tip).

**Status (2026-10-28):**
- T1, T2, T3, T5 and T6 ran on today's traces (results in §2).
- T7 passed a mock-cell run: 184 actions, 0 rejected, hovers at 80/40 mm and
  R = 0/−90 recorded, an out-of-workspace false detection skipped.
- T4 and the T3 click/triangulation path were exercised on the mock trace for
  plumbing only (the mock has no kinematics, so those numbers mean nothing).
- T8 recovered planted errors from synthetic data: a +12.0 mm rim plane, and a
  tool-frame offset of (0, 3.9, −19.9) mm against a planted (0, 4, −20)
  (`evidence/t8_synthetic_selftest.csv`).
- Nothing has been run on the real cell yet.

**Before the T7 slot:**
- The cell serves only the cameras declared in the approved submission. Set
  `Policy.CAMERAS = ["table", "side", "wrist"]` in
  `PhysicalAI-Hackathon-2026/policy.py` before `run.py --submit`.
- Keep `--grasp-check` for a supervised slot; it is the only step that touches
  anything.
- The probe refuses hovers below 20 mm and waypoints outside the UF0 limits.

---

## 5. Workflow

```mermaid
flowchart TD
    A["Misses: vials left, caps right, plus height"] --> B["Step 0: measure each miss from the side camera or a ruler<br/>left/right = robot ∓y, toward/away = robot ±x; note the wrist pose: vertical or cone"]
    B --> B2{"Left/right miss only in cone-press runs?"}
    B2 -- "yes: suspect tool length H2" --> J
    B2 -- no --> C{"Does the miss only exist in the TableCam image?"}
    C -- yes --x> C1["Height-only error seen through perspective<br/>vanishing column u≈906. Go to the height path"] --> D
    C -- no / unsure --> D["Step 1: ruler at the cell<br/>rim height, spare-cap top on the sheet, sheet thickness, tray pitch x/y"]
    D --> E["T2: tray pitch → plane height<br/>T6: depth heights"]
    E --> F{"Do x and y pitch agree<br/>within 5 mm?"}
    F -- no --> G["T5: compare with the earliest frame / LeRobot dataset"]
    G --> H{"Camera moved or tilted?"}
    H -- yes --> H1["Re-solve the TableCam pose with ArUco<br/>keep fx/k1 from cameras.json"] --> E
    H -- no --> H2["Real tray ≠ twin: use the measured pitch and heights"] --> I
    F -- yes --> I["Set --work-surface-z-m / --vial-rim-z-m from the ruler<br/>remember: this also moves XY"]
    I --> J["T7 probe slot: non-contact hovers at 40 and 80 mm, R 0 and −90<br/>left / centre / right tray + spare caps; table + side + wrist + joints"]
    J --> K["T4: FK vs reported TCP"]
    K --> L{"UF0 / UTOOL1 consistent?"}
    L -- no --> L1["Fix the frame assumptions with the cell operator"] --> J
    L -- yes --> M["click_points + T3: triangulate real fingertips and features"]
    M --> N["T8: fit localization L0/L1/L2 and tool T0/T1/T2"]
    N --> O{"Which model wins?"}
    O -- tool-frame offset T1 --> O1["Re-teach UTOOL1 or set pick/place/press offsets from tool z;<br/>re-measure --cone-tip-from-tcp-m"]
    O -- plane L1 --> O2["Correct the per-class planes; spare caps include the sheet"]
    O -- rotation L2 --> H1
    O -- per-class residual --> O3["Detection-centre bias: per-class plane or retrain on real frames"]
    O1 --> P
    O2 --> P
    O3 --> P
    P["T7 --grasp-check, supervised: image the held cap"] --> Q{"Cap centred in the fingers?"}
    Q -- no --> Q1["Grasp transfer H7: fix the pick XY first;<br/>the place error is the mirror of the pick error"] --> J
    Q -- yes --> R["Re-run the T7 hovers: tip-target error < 2 mm everywhere?"]
    R -- no --> N
    R -- yes --> S["Contact trials: offsets +10 → +5 → +2 mm as in README"]
```

---

## 6. Recommended order of work

1. **Check which runs showed "vials left"** and whether they used
   `--crooked-action cone-press` (`policy_config.json` in each run). Also note
   whether those misses had a toward/away component. This takes five minutes
   and picks between H2 and H1.
2. **Measure with a ruler.** It's free and settles the E6-vs-E7 contradiction.
   Record:
   - the rim height above the table;
   - the spare-cap top height on the pink sheet;
   - the sheet thickness;
   - the tray pitch toward/away from the camera (T2 `--pitch-x-mm`, robot x) and
     left/right (T2 `--pitch-y-mm`, robot y).
3. **Rerun T2 with the measured pitch.** Set the planes from the ruler. Then
   rerun T1 to see how far each target moves.
4. **One T7 slot.** About ten minutes of hovering covers H1–H5. Add
   `--grasp-check` only if supervised.
5. **Fit with T8.** Apply the winning correction, then re-hover to verify before
   any contact run.

---

## 7. Notes

- `src/remediation/controller.py` has uncommitted edits (408 lines, saved
  16:37 today) that are not part of this work. This report cites function names
  rather than line numbers for that reason.
- Hypotheses H2 and H7 are about the tool and the grasp, not the camera. The
  evidence (E1–E4) says the camera, base frame and tool *direction* line up, so
  the remaining errors are most likely in the fingertip definition and the
  height planes rather than in `cameras.json`.
