# Innovation Lab Fall 2026: Team Project Plan

## 1. Executive Summary & Core Mission
**Mission:** Maximize total throughput of non-defective, fully functional products on a high-mix, low-volume manufacturing line by combining vision defect inspection with closed-loop robotic rework and sorting.

**Key Design Philosophy:**
- **Non-Invasive / Modular Integration:** "Slapping an edge camera and an auxiliary robotic arm onto an existing line should be enough." The system must adapt to brownfield manufacturing facilities with minimal hardware modification and low capital expenditure.
- **In-Context Generalizability:** Leverage synthetic data (Isaac Sim / Omniverse Replicator) and in-context learning to rapidly reconfigure inspection and robotic manipulation for new vial types, cap geometries, and recipe mixtures without requiring weeks of re-programming.
- **Closed-Loop Action Over Passive Flagging:** Rather than just logging bad vials to a dashboard, actively intervene to fix defects (re-seat caps), discard damaged components to prevent line collisions, and backfill fixture slots to achieve 100% fixture yield.

---

## 2. Problem Statement & Scope

Modern pharmaceutical and chemical fill lines suffer significant downtime and scrap waste due to three cascading issues:
1. **Mechanical Vial & Cap Defects:** Crooked caps, missing caps, cracked glass, or unexpected empty fixtures.
2. **Recipe & Fluid Discrepancies:** Incorrect mixing ratios, fluid volume underfill/overfill, and recipe carryover contamination.
3. **Line Jamming & Yield Loss:** Defective or broken vials that pass unaddressed cause robotic pick collisions downstream, resulting in broken glass, line stoppages, and manual cleanup.

### Target Outcomes
- **Throughput Optimization:** Intercept and repair defects without stopping the main conveyor line.
- **Zero-Collision Assurance:** Safely isolate and clear defective/broken vials before fixture transit to downstream stations.
- **High Fixture Density:** Backfill empty or discarded slots from a buffer tray of verified good vials.

---

## 3. System Architecture & Technical Approach

```
  +-------------------------------------------------------------+
  |              1. Vision & Defect Detection                   |
  |  - FillCam / CapSealCam (CSI Digital Twin + Edge Cameras)   |
  |  - Synthetic Data Training via Replicator (Domain Rand)     |
  |  - Detects: Missing caps, tilted caps, cracks, liquid level |
  +------------------------------+------------------------------+
                                 |
                                 v
  +-------------------------------------------------------------+
  |               2. Decision & Planning Agent                  |
  |  - Severity Classification: Fixable vs Scrap vs Purge       |
  |  - Collision avoidance & Tray synchronization logic         |
  |  - In-context adaptation for new vial/cap types             |
  +------------------------------+------------------------------+
                                 |
         +-----------------------+-----------------------+
         |                                               |
         v                                               v
+-------------------------------+   +-----------------------------------+
|  3. Robotic Remediation       |   |  4. Station Fluid Control         |
|  - Standalone FANUC Arm       |   |  - Pump logic adjustment          |
|  - Re-seat / tighten caps     |   |  - Line purge sequence on         |
|  - Pick & discard scrap vials |   |    recipe carryover               |
|  - Backfill slot from buffer  |   +-----------------------------------+
+-------------------------------+
```

### Stage 1: Multi-Modal Defect Detection
- **Synthetic Training Pipeline:** Generate diverse synthetic datasets across varying lighting, camera angles, bottle fill levels, and cap misalignments using Isaac Sim 6.1 (`RayTracedLighting` / Replicator).
- **Detection Targets:**
  - *Caps:* Missing cap, loose/misaligned cap, crooked seal.
  - *Vials:* Cracked/broken glass, absent vial (empty slot).
  - *Fluid/Recipe:* Liquid height, discoloration / recipe contamination.

### Stage 2: Robotic Intervention & Sorting (FANUC ER-4iA)
- **Rework Hierarchy:**
  1. **Fixable (Cap Misalignment):** Attempt mechanical re-press or cap replacement.
  2. **Irrecoverable (Broken Glass / Contaminated Fluid):** Pick and divert to scrap chute. If glass integrity is compromised, flag safety protocol.
  3. **Empty / Cleared Slot:** Pick a pre-inspected good vial from a 10-unit replacement buffer and populate the fixture.
- **Collision Avoidance:** Timing logic to service the fixture either during dwell time at the inspection station or in an off-line staging buffer to avoid interrupting the primary conveyor timing.

### Stage 3: Closed-Loop Pump & Recipe Control
- If fluid inspection detects recipe drift or cross-batch carryover, trigger a pump purge command to prevent further contaminated vials from entering the line.

---

## 4. Hardware, Compute & Resource Strategy

| Resource | Environment | Purpose |
| :--- | :--- | :--- |
| **Cluster Compute** | MSOE Rosie (Tesla T4 nodes via Slurm) | Batch synthetic data generation via Isaac Sim 6.1 (`scripts/submit_isaac_job.slurm`). |
| **Dedicated GPU Node** | MSOE Dell GB10 (`spark3.ai.msoe.edu`) | Fast interactive testing, real-time Replicator synthesis, model training & inference. |
| **Cloud GPU (Backup)** | Brev.dev (NVIDIA Launchables) | Scalable training runs and model checkpoint validation. |
| **Physical Assets** | UWM Connected Systems Institute (CSI) | Real-world validation against physical vial inspection line and FANUC robot arm. |

---

## 5. Phased Implementation Roadmap

### Phase 1: Synthetic Data Generation & Defect Detection (Week 1–2)
- [x] Configure headless Isaac Sim 6.1 on Rosie and GB10.
- [x] Validate CSI twin renders (`FillCam`, `CapSealCam`) and FANUC scene generation (`runs/fanuc_scene.usd`).
- [ ] Build Replicator randomization script: variations in cap angle (0° to 45° tilt), missing caps, fill levels, and lighting.
- [ ] Train edge object detection / segmentation model (e.g., YOLOv11 / RT-DETR) on synthetic data.
- [ ] Validate detection accuracy against CSI real testbed baseline images.

### Phase 2: Robotic Manipulation & Remediation (Week 2–3)
- [ ] Define pick-and-place trajectories in the FANUC simulation (`fanuc_er4ia_sim`).
- [ ] Implement decision logic: `Scrap` vs `Re-seat Cap` vs `Backfill Empty Slot`.
- [ ] Test collision-free motion planning with moving fixtures.

### Phase 3: Closed-Loop Integration & Evaluation (Week 3–4)
- [ ] Connect vision inference output to robot action trigger.
- [ ] Prototype pump purge logic for recipe contamination.
- [ ] Quantify metrics: Fixture yield improvement, defect detection recall, and simulated throughput gain.
- [ ] Final demonstration package & mentor presentation.

---

## 6. Team Plan Submission Template (For All-AI-Network)

*Copy/paste the following directly into the [Team Plan Submission Form](https://dashboard.all-ai-network.org/s/V28NHK%3Fd%3D1ff0683a-39d7-4302-a363-0835ca390c98):*

- **Team Name & Members:** Team 14
- **Dataset(s) Selected:** Dataset 1 (Final Vial Inspection) + Dataset 3 (FANUC Grasping and Sorting)
- **Problem Summary:** Developing a low-cost, generalizable closed-loop defect detection and remediation cell. An edge camera identifies cap/vial/fluid defects, and an auxiliary robotic arm sorts defective vials, attempts cap remediation, and backfills missing slots from a buffer to ensure 100% fixture output yield without line interruptions.
- **Additional Data Needed:** Physical camera lens calibration parameters and sample real-world fixture dwell times from the CSI testbed.
- **Compute Resources Requested:** MSOE Dell GB10 (`spark3.ai.msoe.edu`) + Rosie HPC Slurm cluster + Brev credits for model training.
- **AI Agent Tools:** Antigravity / Cursor coding agents with NVIDIA Omniverse & PyTorch libraries.
- **Mentor Support Requested:** Guidance on practical collision-avoidance timing with conveyor fixtures and Rockwell FactoryTalk/Emulate3D integration patterns.

---

## 7. Generalizable Defect Detector Trainer

### Objective

Build a configurable real-image training pipeline that can create a YOLO defect detector from a small human-reviewed seed set, LocateAnything object proposals, and VLM-assisted crop labels. The resulting real-image dataset can train a detector directly or fine-tune a checkpoint pretrained on synthetic data.

The trainer is an extension of the inspection and dataset tooling. It must not replace or introduce an alternative to the compliant remediation runtime described in `AGENTS.md`.

For vial-cap inspection, the learned appearance classes are:

1. `good_cap`: an intact cap, whether seated on a vial or lying on the table.
2. `crooked_cap`: a cap visibly tilted, raised, or off-center on a vial.
3. `missing_cap`: an open vial top where a cap should be present.

`table_cap` is not a learned appearance class in new datasets. At runtime, calibrated RGB-D height determines whether a detected `good_cap` is seated on a vial or is a spare on the table. A compatibility mapping may continue emitting `table_cap` to the existing remediation controller until its observation schema is updated.

### End-to-End Workflow

1. Define the object prompt, defect taxonomy, crop scale, review thresholds, split policy, and training settings in a project configuration file.
2. Group and split real captures by video, tray state, specimen, or capture session before generating weak labels.
3. Manually draw boxes and assign appearance labels on a small, diverse training seed set.
4. Expand each seed box to 1.25 times its width and height, producing reviewed VLM reference crops.
5. Run LocateAnything over the remaining full images using a class-agnostic object-of-interest prompt.
6. Create 1.25-times context crops from the generated proposals while retaining the original proposal boxes.
7. Use a few-shot VLM to assign an appearance class or `uncertain` to every crop.
8. Send low-confidence, conflicting, malformed, or anomalous results to human review.
9. Attach the accepted class to the original, unpadded full-image box.
10. Export versioned YOLO train, validation, and test splits with complete provenance.
11. Train from a standard pretrained YOLO checkpoint, fine-tune a synthetic checkpoint, or train with controlled real/synthetic replay.
12. Evaluate against a fully human-reviewed real holdout before registering or deploying the checkpoint.

### Project Configuration

The reusable implementation must obtain class names and behavior from configuration instead of importing the remediation-specific `CLASS_NAMES`. A vial-cap configuration should express at least:

```yaml
schema_version: 1
project_name: vial_cap_defects

localization:
  model: nvidia/LocateAnything-3B
  prompt: "cap or open top of each vial"
  nms_iou: 0.70

classes:
  - name: good_cap
    description: "Intact cap with no visible damage or deformation."
  - name: crooked_cap
    description: "Cap on a vial that is tilted, raised, or off-center."
  - name: missing_cap
    description: "Open vial top where a cap should be present."

crop:
  scale: 1.25
  square: false

classification:
  uncertain_label: uncertain
  automatic_acceptance: 0.90
  review_threshold: 0.70
  require_consensus: true

split:
  group_key: capture_sequence
  train: 0.70
  val: 0.15
  test: 0.15
```

A crop scale of `1.25` means 25 percent greater total width and height, centered on the source box and clipped to the image boundary. The expanded box supplies VLM context only; it must never replace the source box in YOLO labels.

### Canonical Object Record

Every human or generated proposal receives a stable object ID and a manifest record containing:

- Source image, image dimensions, split, and split group.
- Original full-image box and padded crop box in pixel coordinates.
- Localization model, prompt, raw response, and proposal provenance.
- Appearance label, label source, confidence, VLM response, and reference IDs.
- Review status and any human correction.

The original box is authoritative for YOLO geometry. The padded box is authoritative only for crop extraction. Human corrections override the effective label but do not erase the original model response.

### Dataset Splitting and Leakage Control

- Assign splits before VLM classification.
- Keep related video frames, tray states, specimens, and capture sessions in one split.
- Use only training-split seed crops as VLM references.
- Make the validation set predominantly human-reviewed.
- Make the test set fully human-labeled and never use it for prompt selection, references, threshold selection, or training.
- Detect near-duplicate frames and prevent them from crossing split boundaries.

### LocateAnything Proposal Stage

Generalize the current resumable LocateAnything inference code so it accepts unlabeled image directories and a configurable object prompt. The stage must:

- Produce class-agnostic full-image boxes.
- Preserve raw responses and inference configuration.
- Clamp invalid coordinates and discard degenerate boxes.
- Merge duplicate proposals with configurable NMS or clustering.
- Resume at image granularity.
- Record images with zero proposals and unusual object counts.
- Generate proposal overlays for audit.

LocateAnything should locate objects, not decide whether they are good or defective. Separate class-specific localization prompts should be reserved for evaluation experiments rather than the default trainer.

### Crop and VLM Classification Stage

Crop generation must be deterministic and record boundary clipping. The VLM receives class definitions, several reviewed training examples per class, and one candidate crop. It must return structured output such as:

```json
{
  "label": "crooked_cap",
  "confidence": 0.91,
  "reason": "The cap is raised and tilted relative to the vial rim."
}
```

The classifier must reject unknown classes and malformed responses and must support an explicit `uncertain` result. The preferred initial policy is:

- Confidence at least `0.90` with classifier consensus: automatically accept.
- Confidence from `0.70` through `0.90`: require human review.
- Confidence below `0.70`, disagreement, malformed output, or `uncertain`: exclude until reviewed.

Consensus may combine direct multiclass classification with a second reference-similarity pass. The VLM backend must be replaceable without changing crop generation or YOLO export.

### Review and Dataset Finalization

The review output must include:

- Low-confidence and conflicting predictions.
- Heavily clipped, unusually small, or unusually large boxes.
- Images with zero or unexpected proposal counts.
- Per-class crop contact sheets.
- Full-image overlays using final boxes and labels.
- A random audit sample of automatically accepted labels.

Finalization writes YOLO `images/<split>` and `labels/<split>` directories, `data.yaml`, a dataset manifest, a build report, and preview overlays. Reviewed negative images require empty label files so that they contribute valid background examples.

### Synthetic Data and Training Modes

The trainer must support:

1. **Real only:** train a standard pretrained YOLO model from accepted real annotations.
2. **Synthetic initialization:** fine-tune the best synthetic detector on accepted real annotations. This is the preferred initial vial-cap mode.
3. **Mixed replay:** train with real annotations plus a controlled, reproducible sample of synthetic examples to reduce catastrophic forgetting.

New three-class synthetic datasets are derived without modifying the original four-class datasets:

```text
old good_cap  -> good_cap
old table_cap -> good_cap
old crooked_cap -> crooked_cap
old missing_cap -> missing_cap
```

Training records the initial checkpoint, dataset and configuration hashes, class mapping, hyperparameters, per-class metrics, and real and synthetic holdout results. GPU VLM inference and YOLO training must use Slurm. Run artifacts belong under `shared/runs/YYYY-MM-DD_<project>_weak_labeling/`.

### Runtime Z Qualification

The detector predicts appearance only. For each detected `good_cap`, sample valid depth inside an eroded region of its box and compute a robust height statistic rather than using one center pixel.

Expected top heights are:

```python
seated_z = rim_z + SPEC.flange_thickness
spare_z = surface_z + SPEC.plug_depth + SPEC.flange_thickness
```

Apply configurable, empirically calibrated tolerances:

```text
Z near seated_z -> seated_good_cap
Z near spare_z  -> spare_cap
invalid or intermediate Z -> ambiguous
```

No backprojected aperture or vial XY coordinates are required for this gate because the table and vial-top states differ by approximately the vial height. The depth calculation must ignore invalid pixels, require a minimum valid-pixel count, and reject ambiguous observations. During migration, `spare_cap` maps to the controller's current `table_cap` observation; ambiguous caps are never offered to the controller as spares.

### Evaluation and Release Gates

Evaluate each stage separately:

- **Localization:** proposal recall, IoU, center error, missed-object rate, and duplicate rate.
- **VLM labeling:** reviewed accuracy, confusion matrix, per-class precision and recall, abstention rate, confidence calibration, and human correction rate.
- **YOLO detector:** real-image precision, recall, mAP, per-class failures, capture-condition performance, and synthetic regression.
- **Z qualification:** seated-versus-table accuracy, invalid/ambiguous rate, false-spare rate, and missed-spare rate.

Robot release requires a fully reviewed operational evaluation set and zero false-spare selections. A detector checkpoint remains experimental until real-image and contact-localization gates pass.

### Required Tests

- Exact 1.25-times crop expansion and image-boundary clipping.
- Preservation of original proposal geometry through classification and export.
- YOLO coordinate round trips.
- Stable object IDs and resumable inference.
- Rejection of malformed, unknown, or low-confidence VLM results.
- Group-aware splits with no reference examples from validation or test.
- Empty-label handling for reviewed negative images.
- Four-class to three-class synthetic label conversion.
- Robust depth sampling and seated/table/ambiguous Z decisions.
- Compatibility conversion from a table-height `good_cap` to the existing `table_cap` controller observation.

### Delivery Phases

#### Phase A: Weak-Labeling MVP

- Add configuration, schemas, seed import, proposal generation, crop generation, VLM classification, review reports, and YOLO export.
- Validate on a small, fully reviewed subset before processing the complete real dataset.

#### Phase B: Detector Training

- Add three-class synthetic conversion, real/synthetic composition, YOLO fine-tuning, Slurm launchers, and real holdout evaluation.

#### Phase C: Runtime Integration

- Add robust depth sampling, Z-based seated/spare qualification, compatibility mapping, controller tests, and end-to-end dry runs.

#### Phase D: Active Learning

- Run the trained detector on additional real captures.
- Prioritize uncertain, conflicting, and visually novel examples for review.
- Add corrections to the versioned seed set and retrain.
- Track whether each iteration reduces VLM abstentions and human correction rates.

