# Official Physical-Robot-Parity Remediation Videos

Every video in this directory demonstrates the **exact, uncheated physical robot remediation process** as executed on the physical Fanuc ER-4iA workcell (`PhysicalAI-Hackathon-2026/policy.py`).

## The Physical Robot Process (Zero Cheating / Pure Compliance)
None of these videos use artificial snapping, kinematic teleportation, table regrip tricks, or shortcut heuristics. Every run strictly follows the full, closed-loop physical pipeline:
1. **Overhead Scan**: Calibrated TableCam takes a planning snapshot; YOLO26x detects coarse defect targets and TableCam depth fits the work-surface plane.
2. **Missing Cap Remediation First**:
   - Robot approaches the coarse spare cap position at tray clearance ($> 0.26\text{ m}$).
   - Descends to 300 mm optical distance for wrist camera inspection.
   - NVIDIA LocateAnything-3B foundation model refines the millimeter-accurate loose cap bounding box.
   - Low horizontal slide approach (`pickup_slide_x_m = -0.040 m`): arm descends 40 mm back, slides into the cap pocket at pickup height, and closes the conformal jaws around the flange.
   - Lifts cap to safe transit clearance.
   - Transits over the missing vial mouth, descends for wrist camera refinement of the circular opening.
   - Lowers plug 15 mm into the mouth, releases gripper, executes jaw clearance moves (`side_step = 1 mm`, `exit_step = 2 mm`), and retreats.
3. **Crooked Cap Remediation Second**:
   - Transits over the crooked cap, descends for wrist camera refinement.
   - Cable-safe orientation roll: arm stays vertical while joint J6 executes a clockwise 90° roll.
   - Leans arm into the nominal 45° TPU cone press posture (`cone_press_wpr_deg = [135, 0, -90]`).
   - Descends along the cone axis, presses the cap flush on the rim with bounded force, and retreats.
4. **Post-Remediation Verification**: Arm returns to home inspection pose and confirms all vials are intact and seated flush.

---

## Video Catalog

| File | Perspective | Duration / Size | Workflow Details |
|---|---|---|---|
| **`full_workflow_tablecam_yolo.mp4`** | Overhead TableCam | 1:15 (7.5 MB) | Complete verified physical workflow with live YOLO bounding box overlays and state transitions. Missing cap seated flush, crooked cap pressed flush, all vials intact. |
| **`full_workflow_sidecam_diagnostic.mp4`** | Orthogonal SideCam | 1:15 (6.8 MB) | Synchronized side perspective of `full_workflow_tablecam_yolo.mp4`, clearly showing the wrist camera inspection descents, horizontal slide pickup, and angled cone press contact. |
| **`physical_robot_workflow_tablecam.mp4`** | Overhead TableCam | 1:18 (6.5 MB) | Complete 4,676-frame continuous execution of the physical match simulation (`2026-10-31_physical_match_sim`), verifying full pipeline timing. |
| **`physical_robot_workflow_sidecam.mp4`** | Orthogonal SideCam | 1:18 (3.2 MB) | Synchronized side view of the complete 4,676-frame physical match simulation run from home to final verification. |
| **`multi_target_tablecam_yolo.mp4`** | Overhead TableCam | 1:55 (11.0 MB) | Extended multi-target batch remediation with 7 separate wrist-camera LocateAnything VLM refinement snapshots. |
| **`multi_target_sidecam_diagnostic.mp4`** | Orthogonal SideCam | 1:55 (5.3 MB) | Diagnostic side view of the extended multi-target batch remediation, demonstrating consistent transit clearance and contact compliance across multiple slots. |
| **`cone_press_batch_tablecam.mp4`** | Overhead TableCam | 0:54 (4.9 MB) | Dedicated batch run showing the TPU side-cone tool pressing multiple crooked caps sequentially without redundant wrist re-orientations. |

All videos are encoded with H.264 (`libx264` / `yuv420p` / CRF 24–26) for universal playback in PowerPoint, Google Slides, Keynote, QuickTime, VLC, and HTML5 video tags.
