# Architecture & Operations Guide: Two-Tier Vial Inspection System

**Project:** Rockwell Automation + NVIDIA Innovation Lab Fall 2026  
**Target:** Dataset 1 (Final Vial Inspection) & Dataset 3 (Robotic Remediation)  
**Host Environment:** Dell GB10 Grace Blackwell ARM64 (`sm_121`, 121 GB unified memory)

---

## 1. Executive Summary: The Two-Tier Paradigm

To achieve the twin goals of **ultra-low-cost, sub-millisecond edge inspection** and **automated zero-cost labeling for high-mix lines**, the system implements a two-tier architecture:

```
                      +------------------------------------------+
                      |         Real Factory Overhead Video      |
                      |   (shared/data/csi_vial_fill_dataset)    |
                      +--------------------+---------------------+
                                           |
                                           v
+------------------------------------------+------------------------------------------+
|  TIER 1: Real-Time Edge Inspection       |  TIER 2: Offline Data Engine / VLM       |
|  (In-Line Brownfield Line)               |  (Active Learning & Zero-Cost Labeling)  |
+------------------------------------------+------------------------------------------+
| - Model: YOLOv11-cls + Golden Delta      | - Model: Cosmos Reason / SmolVLM         |
| - Cycle Time: <15 ms for all 30 slots    | - Method: Visual In-Context Prompting    |
| - Compute: Ultra-lightweight (3.2 MB)    | - Task: Auto-labels real images (1,181+) |
| - Output: Robotic Remediation Commands   | - Output: Discovers rare edge cases      |
|   (`press_reseat`, `scrap_and_backfill`) |   and retrains Tier 1 edge model         |
+------------------------------------------+------------------------------------------+
```

---

## 2. Tier 1: Real-Time Edge Inspector (Production Line)

### Key Files:
- Aperture Grid Calibration: [`../src/inspection/config/fixture_g_apertures.json`](../src/inspection/config/fixture_g_apertures.json) (Good) & [`../src/inspection/config/fixture_s_apertures.json`](../src/inspection/config/fixture_s_apertures.json) (Scrap).
- Inspector Engine: [`../src/inspection/aperture_inspector.py`](../src/inspection/aperture_inspector.py)
- Trained Edge Model: [`../shared/inspection_models/vial_aperture_cls_best.pt`](../shared/inspection_models/vial_aperture_cls_best.pt)
- Golden Reference: [`../shared/inspection_models/golden_reference.png`](../shared/inspection_models/golden_reference.png)

### Specifications:
- **5 Official Competition Classes**: `pass`, `crooked_cap`, `missing_cap`, `damaged_vial`, `no_vial`.
- **Ensemble Logic**:
  1. **Golden Delta (Visual ChangeNet Baseline)**: Compares each slot crop directly to a verified nominal reference. Nominal delta is $\le 4.7$; any defect produces $\Delta \ge 10.0$ (up to $48.3$ for missing caps).
  2. **YOLOv11-cls Neural Verification**: Evaluates 32x32 aperture crops in batch on the GPU in $\sim 0.5\text{ ms}$ per slot, providing multi-class probability distributions.
  3. **Fill-Order Sequence Context**: Tracks the robot's column fill order (right-to-left, top-to-bottom) to distinguish scheduled empty positions (`pending_fill`) from missed robot drops (`emergency_missed_placement`).

### Remediation Action Mapping:
| Detected Class | Decision Action | Robot Arm Command |
|---|---|---|
| `pass` | `pass` | None (keep on line) |
| `crooked_cap` | `open_shoulder_reseat` | TableCam-targeted open-claw descent; CAD internal shoulder presses cap; position-limited, no force input |
| `missing_cap` | `pick_place_press` | Pick nearest camera-detected loose cap $\to$ place $\to$ position-limited press |
| `damaged_vial` | `scrap_and_backfill` | Safe gripper retraction $\to$ scrap $\to$ buffer pick |
| `no_vial` (missed) | `emergency_missed_placement` | Immediately pick replacement from buffer to backfill |

---

## 3. Tier 2: Offline In-Context VLM Auto-Labeler (Data Engine)

### Key Files:
- VLM Labeler: [`../src/inspection/vlm_auto_labeler.py`](../src/inspection/vlm_auto_labeler.py)
- Real Dataset Source: `../shared/data/csi_vial_fill_dataset/cap_seal/images/` (1,181 captures)

### Methodology (In-Context Reasoning):
Instead of paying human operators to annotate thousands of frames for each recipe changeover:
1. Multi-shot visual exemplars (`pass`, `crooked_cap`, `missing_cap`, `no_vial`) are constructed from the digital twin.
2. The VLM is prompted with the reference exemplars and the real query crop.
3. The VLM reasons through physical cues (yellow cap orientation, thread elevation, glass refraction, fixture lip visibility).
4. If confidence $\ge 0.90$, the label is auto-accepted into the training dataset. Borderline anomalies are flagged for human operator review.

---

## 4. Execution Commands (Using `scripts/*.sh`)

### Run All (Full End-to-End Pipeline in One Command):
```bash
./scripts/run_all.sh
```
This executes the full sequence automatically:
1. Renders clean nominal golden reference (`golden_reference.png`).
2. Generates procedural synthetic defect crops via Isaac Sim digital twin.
3. Trains the YOLOv11-cls neural model on the GPU (100% Top-1 acc).
4. Runs real-time edge inspection and outputs structured remediation decisions.

---

### Run Individual Modules:
```bash
# Run with default golden reference & test defect scene
./scripts/run_inspection.sh

# Or run on any custom image
./scripts/run_inspection.sh --image /path/to/capture.png
```

### Render Clean Golden Reference Image:
```bash
./scripts/render_golden.sh
```

### Run Synthetic Twin Data Generation:
```bash
./scripts/run_data_generation.sh --scenes 10
```

### Re-Train YOLOv11 Edge Classifier:
```bash
./scripts/run_train_classifier.sh --epochs 25 --batch 16
```

### Run Offline VLM Auto-Labeler on Real Images:
```bash
./scripts/run_vlm_labeler.sh --max-images 5
```
