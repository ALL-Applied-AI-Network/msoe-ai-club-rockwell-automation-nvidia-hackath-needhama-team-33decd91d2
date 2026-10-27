# Real-Life Demo: What Can Go Wrong, and the Prepared Fixes

Everything in the remediation pipeline has so far run only in Isaac Sim
(`physical_robot_authorized: false` in every attempt report). This list covers
what is likely to break when the same pipeline drives the real Fanuc ER-4iA,
SMC gripper, TableCam and wrist camera, with a prepared fix for each. Items are
ordered by how likely they are to stop the demo.

Each fix is marked as one of:
- **[in code]**: already implemented. The fix is a parameter to tune.
- **[ready]**: not implemented yet. The change and where it goes are spelled out.
- **[procedure]**: done at the venue, no code change.

---

## 1. Calibration

### 1.1 Tool centre point (TCP) and pocket-ledge offset are not measured
*Risk:* the tool offsets come from CAD/simulation only. `PHYSICAL_SPECIFICATIONS.md`
section 4 says the 284.3 mm CAD envelope and the 100 mm provisional offset must not be
used for contact. A 1 mm error in z makes the press either miss the cap or push
1 mm into the glass. A 1 mm error in xy makes the plug hit the rim (the plug top has
only 0.13 mm radial clearance).

*Fix [procedure]:* before the demo:
1. Teach the TCP with the Fanuc 4-point/6-point method on the closed fingertips.
2. With a cap gripped, touch the flange underside to a gauge block. That distance is
   `cap_seat_tcp_j6_m` (the pocket ledge). Measure the closed fingertip face
   (`finger_tip_tcp`) the same way.
3. Enter both in `custom_gripper.py` (mount values returned by `attach_visual_pair`).
4. Check: jog the fingertip face to `rim_base_z() + flange_thickness` over a seated
   cap. It should just touch, within 0.3 mm.

### 1.2 TableCam extrinsics are off, or the tray is rotated
*Risk:* the tray-grid fit finds translation only (`fit_tray_center`). In sim the
tray yaw is 0; a hand-placed real tray will be rotated. At the far slots, 1 deg of
yaw is about 2.6 mm of error, which makes the plug miss the mouth.

*Fix [ready]:* add yaw to `fit_tray_center`. Search yaw from -3 to +3 deg in 0.25
deg steps around the depth-measured plate centre, rotating `slots` about the
origin before translating, and keep the lowest score. The search is about 25 x
400 points, which is cheap. Return the yaw and apply it in `slots()`.

*Check [in code]:* `tray_fit_error_m` is printed and saved in the attempt report.
Above 2.5 mm, stop and re-run the TableCam extrinsic calibration (checkerboard on
the plate) before moving the arm.

### 1.3 Vials stand off-grid
*Risk:* each aperture has 5.08 mm diametral clearance, so real vials lean or shift by
up to 2.5 mm. The runner places caps at the fitted grid position, not at the
measured vial.

*Fix [ready]:* for missing slots, use the centroid of the vial rim-ring depth points
(the points `classify_slots_from_depth` already counts as `rim_pts`) as the target
xy, provided there are at least 60 points. Otherwise keep the grid. For extra
margin, push the vials to one side of their apertures with a foam strip so they
all lean the same way.

---

## 2. Perception

### 2.1 Real depth does not see clear glass
*Risk:* the sim's ray-traced depth returns glass as a surface. Real stereo or
time-of-flight depth usually returns nothing, or the background, on clear glass
rims. The new depth slot check (`classify_slots_from_depth`) relies on rim points:
with none, real missing caps are classified `empty` and dropped, and real crooked
caps fall back to YOLO.

*Fix [ready]:*
1. Only let the depth check veto a missing-cap detection (`empty`) when vials are
   visible in depth at all, i.e. when any loaded slot shows at least 60 rim points.
   Otherwise trust YOLO for `missing_cap`.
2. Keep the depth check that adds crooked caps. It uses cap surface (`above_bore_pts`),
   which is opaque silicone and shows up in real depth.
3. As a last resort, add an operator-supplied `--loaded-slots G01,G02,...` list
   that marks which apertures hold vials.

*Check [procedure]:* capture one TableCam depth frame at the venue and run
`classify_slots_from_depth` offline (as in `shared/runs/2026-10-27_*` checks) before
the demo.

### 2.2 The YOLO checkpoint does not transfer to real images
*Risk:* `remediation_yolo_best.pt` is trained on sim renders. In sim it already
missed 3-4 of 5 missing caps on 5-defect trays, and labelled one crooked cap as good.
The reviewed real-photo model (`runs/archive/2026-10-26_yolo_reviewed`) does not
transfer to the TableCam view either.

*Fix:*
- **[in code]** The depth slot check covers missed or mislabelled crooked and
  missing caps on the tray. In sim it recovered 30/30 defects against 22/30 from
  YOLO alone.
- **[procedure]** Take 30-50 TableCam photos at the venue with defects in known
  slots. Label them with the reviewed-label workflow (`build_reviewed_capseal_dataset.py`)
  and fine-tune for about 30 epochs (`scripts/training/`). Split train and validation
  by tray state, not by frame.
- **[ready]** Lower the confidence threshold for `table_cap` only. The spares sit on
  plain table, so false positives there are cheap and are rejected by depth (the
  cap top must be at `surface + plug_depth + flange`).

### 2.2b Spare-cap positions from YOLO boxes are biased
*Risk:* in sim, the TableCam box centre of a spare on the table is 1.9-2.6 mm off.
A spare gripped that far off-centre caught its plug on the rim. One was left
1.85 mm high; another rode out on the fingers and was dropped onto a neighbouring
vial, fracturing it.

*Fix [in code]:* spares are located from the centroid of their flange top in depth
(within 0.5 mm in sim). On the real camera, check the flange-top depth of a spare
on the table. Satin silicone gives good depth, but at least 30 points are needed;
otherwise the code falls back to the box centre.

### 2.3 The plate-centre estimate fails on the silver perforated plate
*Risk:* `plate_center_from_depth` needs at least 1000 depth points at plate-top
height. A specular, perforated metal plate can return sparse or noisy depth, which
raises "tray plate not found".

*Fix [ready]:* catch that error and fall back to the unconstrained grid search.
That is safe on a full tray; on a partly loaded tray, also require
`tray_fit_error_m` < 2 mm. Or stick a matte tape strip, or an AprilTag, on two
plate corners and use them as the plate centre.

### 2.4 Robot, gripper or hands in the TableCam view
*Risk:* in sim, a crooked-cap false detection 150 mm off the tray came from the arm.

In sim, the parked gripper also hangs over the back row of the tray in the live
capture. Its depth points stood over an open vial, and a missing cap was read as
crooked.

*Fix [in code]:* `on_tray` drops detections more than 25 mm from any slot, and the
grid fit caps outliers at `FIT_INLIER_M`. The depth slot check ignores anything
more than `SLOT_ABOVE_MAX_M` (40 mm) above the rim. *[procedure]* Park the arm
outside the camera view (or at least away from the tray) during the capture, and
keep hands out of view.

### 2.5 Wrist-camera servo: glare, depth minimum range, wrong normal
*Risk:*
- The ellipse fit on the satin flange ring can lock onto a highlight.
- Depth cameras have a minimum range (often 0.1-0.3 m), and `locate`/servo views are
  0.15-0.22 m from the cap.
- If the depth gradient sign is wrong, the servo picks the mirror normal. In sim
  this happened before the 3 x 3 splat fix.

*Fix:*
- **[procedure]** Use diffuse lighting and lock the wrist camera's exposure. Check
  the depth minimum range against the 0.15 m view distance.
- **[ready]** If the first view's normal is ambiguous (depth gradient below noise),
  take a second oblique view 15 deg toward each candidate normal and keep the
  candidate whose opening gets rounder. `servo_cap_axis` already iterates toward
  the round view, so a wrong first pick shows up as roundness getting worse.

---

## 3. Gripper and contact

### 3.1 The SMC gripper is pneumatic, not position-controlled
*Risk:* the sim commands intermediate jaw positions: `jaw_grip`, `jaw_hold`
(0.4 mm extra clamp, about 11 N per jaw), and `JAW_CLOSED`. The real MHZL2 is
open/closed only, with force set by the supply pressure.

*Fix [procedure]:*
- Set the regulator so grip force is about 10-15 N per jaw. Look up the force
  vs pressure curve in the SMC MHZL2-16 catalogue, and check it with a spring scale
  between the fingers.
- Map `jaw_grip`, `jaw_hold` and `JAW_CLOSED` all to "close".
- The fingertip press uses the closed fingers. Re-measure the closed tip TCP
  (1.1) with the fingers fully closed, not at `jaw_grip`.
- Add a 0.3 s dwell after each valve change. The sim's 20-step jaw ramp is 0.33 s.

### 3.2 The robot has no force sensor, so there is no force-limited press
*Risk:* `Executor.press` stops on vial load measured from PhysX contact reports. The
real cell has no such measurement. The press is position-limited to 0.5 mm below
the seated flange top (`plan_place_and_press`), so any z calibration error becomes
force on the glass.

*Fix:*
- **[procedure]** Measure the real seating force with a kitchen scale under a vial
  before the demo. If seating needs more than about 30 N, the plug taper differs
  from the model.
- **[ready]** Make the press end height a parameter (default `-0.0005`). For the
  first demo runs, start at +0.5 mm and step down 0.25 mm per run.
- **[procedure]** Turn on Fanuc collision detection at high sensitivity for the
  press segments, and run them at 10% speed. Soft Float is better if the controller
  has it.
- **[ready]** Fit a 1-2 mm compliant pad (silicone, Shore 40A) on the fingertip
  face. It turns z error into bounded force.

### 3.3 Real silicone caps deform and stick
*Risk:* the sim caps are rigid with friction 1.2/1.0 (an assumption). Real silicone
can stick to the finger pocket, so the cap is carried off on retreat (seen in sim
for G09), or squeeze out of the flange slot.

*Fix [in code]:* `place_and_press` opens the jaws and holds 40 steps before the
press, and `retreat` lifts with the jaws at `jaw_grip`. *[ready]* If caps ride up,
retreat with the jaws fully open, and add a 5 mm, 0.5 s pause after release before
lifting.

### 3.4 The coaxial regrip knocks the crooked cap into the vial
*Risk:* a real leaning cap is less stable than the PhysX pose (the 48-77 deg band
comes from sim). The open fingers coming down the cap axis can tip it into the vial.
Fishing it back out needs tools.

*Fix:*
- **[in code]** The last `REGRIP_SLOW_M` (20 mm) of the approach runs at
  0.2 mm/step.
- **[ready]** Take one wrist view just before closing, and abort if the opening
  centre has moved more than 2 mm from the servo estimate.
- **[procedure]** Keep tweezers at the cell and pre-agree the "skip this slot"
  narrative.

### 3.5 Cap falls over on the table during the regrasp
*Risk:* the cap is stood on the table and released. With the old wrist-view
regrasp, 2 of 3 G04 runs (cap held 13 deg rocked) toppled while the open fingers
rose past it.

*Fix [in code]:* the regrasp now re-closes in place. The fingers never leave the
cap, so there is no rise for the cap to be dragged on (commit "Regrasp the crooked
cap in place"). *[procedure]* Use a matte, flat surface at the put-down spot,
because a glossy table lets the cap slide. *[ready]* If toppling persists, increase
`PUTDOWN_DROP_M` from -1.5 mm to -2.0 mm so the table squares the cap harder.

---

## 4. Reach and layout

### 4.1 Spares or slots out of reach
*Risk:* in sim, spares at x >= 0.415 m (base frame) and caps near the base that lean
toward the robot (J3 past its -69 deg limit) were unreachable.

*Fix:*
- **[in code]** Each missing slot falls back to the nearest reachable spare, and
  slots left without one are reported `unfilled`.
- **[in code]** The regrip leans the tool up to `REGRIP_MAX_MISALIGN_DEG` (10 deg)
  short of coaxial.
- **[procedure]** Tape the tray outline at the qualified pose (centre 0.305, 0.035 m
  in the base frame). Lay spares in a 3-wide grid at x 0.28-0.37 m, y -0.255 m and
  -0.212 m.

### 4.2 The planner succeeds in sim but the real path collides
*Risk:* the joint-space moves (`s_curve_joints` between waypoints) are checked only
against PhysX. On the real arm, cables and the wrist-camera bracket stick out.

*Fix [procedure]:* do a first dry run with the tray removed and every waypoint
offset +50 mm in z, at T1 (250 mm/s max). Then run the real thing at 30% speed with
a hand on the E-stop.

---

## 5. Integration and timing

### 5.1 Nothing sends the plan to the real robot yet
*Risk:* the runner writes `action/trace.json` with `q_cmd` per 1/60 s step. Nothing
streams it to the controller.

*Fix [ready]:* a small exporter that thins `q_cmd` to about 10 Hz joint waypoints and
sends them over Fanuc RMI (or writes a TP program with J moves and CNT0 for the
contact segments). The gripper open/close comes from the `jaw_m` changes in the
trace. Rehearse it with the tray removed (5.1 and 4.2 go together).

### 5.2 Demo time
*Risk:* each crooked cap takes about 5 sim-minutes (servo, regrip, regrasp, place).
Five crooked caps will not fit a short demo slot.

*Fix [in code]:* missing caps are done first, as quick wins. *[procedure]* Demo
with 1-2 crooked and 2 missing (the 10-vial scenario in
`docs/media/fixed_table_behavior/`), and show the 5+5 video as a recording.

### 5.3 Compute at the venue
*Risk:* YOLO and Isaac Sim need a GPU, and the cluster or network may be unavailable.

*Fix [procedure]:* pre-run perception on the demo laptop's GPU, or on CPU (`--device cpu`
works with the inspection script). Copy the checkpoint and `.conda_env` locally,
and keep the recorded videos in `docs/media/` as the fallback.

---

## Pre-demo checklist (in order)
1. TCP and pocket ledge measured (1.1); touch-off check within 0.3 mm.
2. TableCam calibration; capture one frame; `tray_fit_error_m` < 2.5 mm (1.2).
3. Depth slot check run offline on that frame; vial rims visible or not (2.1).
4. Gripper pressure set; grip force 10-15 N per jaw (3.1).
5. Seating force measured on a vial (3.2).
6. Tray and spares at the taped poses (4.1).
7. Dry run in air, +50 mm z, T1 (4.2).
8. First real run at 30% speed, press end height +0.5 mm (3.2).
