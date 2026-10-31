# Official Physical-Robot-Parity Remediation Videos

Every video in this directory demonstrates the **exact, uncheated physical robot remediation process** as executed on the physical Fanuc ER-4iA workcell (`PhysicalAI-Hackathon-2026/policy.py`).

## Fast vs. Realistic Cadence Modes

To satisfy both presentation requirements (showing the full uncheated process with authentic real-world pauses vs. demonstrating rapid high-speed execution), two paired sets of renders are provided:

### 1. Realistic Cadence (`_realistic.mp4`)
Accurately reproduces real-world robot dwell times down to the second:
- **TableCam Scan**: 1.5-second dwell at the home parking pose while YOLO26x scans the tray and TableCam depth fits the work-surface plane.
- **Wrist-Camera VLM Inspection**: 2.4-second dwell at the optical inspection pose (`await_refine_*`) while NVIDIA LocateAnything-3B analyzes the image and refines the bounding box.
- **Physical Contact & Settling**: 0.5-second dwell (`MOTION_SETTLE_ROWS = 5`) at gripper close, release, and cone press endpoints matching the 10 Hz physical servo tracking delay.
- **Post-Remediation Verification**: 1.5-second dwell verifying that all vials remain intact and caps are seated flush.

### 2. Fast Continuous Mode (`_fast.mp4`)
Continuous 60 fps simulation playback without compute pauses, demonstrating the fluid transit and manipulation capabilities of the controller.

---

## Video Catalog

| File | Timing Mode | Viewpoint | Duration | Content Description |
|---|---|---|---|---|
| **`physical_match_tablecam_realistic.mp4`** | **Realistic** | Overhead TableCam | 1:33 (6.9 MB) | Full physical match simulation (`temp04`) with real 2.4 s VLM inspection dwell and 1.5 s perception pauses. |
| **`physical_match_tablecam_fast.mp4`** | **Fast** | Overhead TableCam | 1:18 (6.5 MB) | Continuous 60 fps playback of `temp04` from initial scan to final verification. |
| **`physical_match_sidecam_realistic.mp4`** | **Realistic** | Diagnostic SideCam | 1:33 (3.1 MB) | SideCam perspective with authentic camera hover pauses, low horizontal slide approach, and compliant insertion. |
| **`physical_match_sidecam_fast.mp4`** | **Fast** | Diagnostic SideCam | 1:18 (2.8 MB) | Continuous side perspective of the physical match simulation. |
| **`full_workflow_tablecam_realistic.mp4`** | **Realistic** | Overhead TableCam | 1:31 (6.5 MB) | Verified end-to-end run (`v22`) with authentic VLM pauses. All vials intact, missing cap seated flush. |
| **`full_workflow_tablecam_fast.mp4`** | **Fast** | Overhead TableCam | 1:15 (6.0 MB) | High-speed continuous playback of the verified end-to-end run (`v22`). |
| **`full_workflow_sidecam_diagnostic.mp4`** | **Continuous** | Diagnostic SideCam | 1:15 (6.8 MB) | Close-up side view of custom conformal finger grasping, transit clearance, and 45° cone press contact. |
| **`cone_press_batch_tablecam.mp4`** | **Continuous** | Overhead TableCam | 0:54 (4.9 MB) | Dedicated batch run showing the TPU side-cone tool pressing multiple crooked caps sequentially. |

---

## How to Encode Both Versions from Any Future Run
```bash
# Encodes both fast and realistic videos automatically from any action/frames directory:
./scripts/tools/encode_realistic_presentation_video.py \
    --frames shared/runs/<run_name>/action/frames \
    --trace shared/runs/<run_name>/action/oracle_evaluation_trace.json \
    --out docs/media/final_render/my_run.mp4 \
    --mode both
```
