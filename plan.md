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

