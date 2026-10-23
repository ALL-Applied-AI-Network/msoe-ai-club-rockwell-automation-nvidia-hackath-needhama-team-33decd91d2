# 🧪 Fluid Mixing Benchmark Report: 200-Vial Dynamic Queue Test
## Rockwell Automation + NVIDIA Innovation Lab Fall 2026 — Team 14
### CSI Station 3 (Fluid Mixing Digital Twin) • Predictive Carryover Minimization & Compensation (PCMC)

---

## 1. Executive Summary

This study evaluates the performance of the CSI Station 3 continuous fluid dispensing testbed across two full-scale 200-vial production tests under **Dataset 2**:
* **Test 1 (Unmodified Algorithm / Baseline):** Simulates existing factory operation without reordering or hold-up compensation, using a sequential FIFO queue and the physical Rockwell PLC dispense sequence (`T200 -> T400 -> T100`).
* **Test 2 (New Algorithm / PCMC Dynamic Orchestrator):** Implements Hamiltonian path batch reordering, intra-vial sequencing, hold-up credit volume deduction ($V_{\text{pump}} = \max(0, V_{\text{target}} - V_{\text{tube}})$), and dynamic mid-production order handling via Option B campaign structuring and Backup Option A boundary priming.

### Dynamic Queue Execution Rules
* **Task Start:** Production begins with **List 1** (100 recipes).
* **Halfway Point (Step 50):** 100 new recipes (**List 2**) are injected dynamically into the active queue.

```
List 1 (Starting Task - 100 recipes):
6, 5, 9, 8, 8, 7, 6, 13, 6, 14, 11, 5, 5, 6, 8, 8, 13, 14, 5, 13, 8, 13, 11, 8, 12, 14, 9, 5, 7, 11, 10, 9, 7, 8, 10, 6, 6, 11, 6, 10, 10, 14, 9, 5, 12, 13, 6, 11, 6, 13, 9, 14, 10, 14, 8, 6, 5, 8, 9, 6, 8, 6, 11, 9, 12, 10, 7, 10, 10, 8, 9, 6, 14, 7, 13, 8, 7, 12, 11, 9, 13, 8, 10, 5, 8, 5, 10, 11, 9, 6, 8, 14, 10, 8, 12, 11, 12, 7, 9, 7

List 2 (Dynamic Mid-Production Injection - 100 recipes):
12, 11, 8, 9, 6, 14, 5, 7, 13, 10, 8, 14, 11, 12, 5, 9, 6, 7, 13, 10, 5, 8, 11, 14, 9, 12, 6, 7, 13, 10, 14, 11, 8, 5, 9, 12, 6, 7, 13, 10, 11, 8, 14, 9, 5, 12, 6, 7, 13, 10, 8, 11, 14, 5, 9, 12, 6, 7, 13, 10, 14, 8, 11, 9, 5, 12, 6, 7, 13, 10, 11, 14, 8, 5, 9, 12, 6, 7, 13, 10, 8, 11, 14, 9, 5, 12, 6, 7, 13, 10, 14, 11, 8, 5, 9, 12, 6, 7, 13, 10
```

---

## 2. Key Performance Indicators (KPIs)

| Metric | Unmodified Algorithm (Baseline) | PCMC New Algorithm (Optimized) | Impact / Improvement |
| :--- | :--- | :--- | :--- |
| **Total Vials Processed** | 200 | 200 | 100% Production Target Met |
| **Total Defective Vials** | **62 / 200 (31.0%)** | **0 / 200 (0.0%)** | **100% Defect Elimination** |
| **Dilution Defects** | 18 vials (Recipe 13 diluted) | 0 vials | 100% Elimination |
| **Color Carryover Scrap** | 2 vials (Recipe 6 contaminated) | 0 vials | 100% Elimination |
| **Spill Violations (>80 mL)** | 42 vials (MagneMover spill risk) | 0 vials | 100% Elimination |
| **Purge Dump Waste** | 1,631.8 mL (if purged between recipes) | **8.2 mL** (Single boundary prime) | **99.5% Waste Reduction** (-1,623.6 mL saved) |
| **Cross-Contamination Vol** | 1,640.0 mL | **0.0 mL** | 100% Pure Chemical Transitions |
| **Reagent Dosing Accuracy** | Off-spec (+8.2 mL carryover) | Exact 100.0% nominal match | Perfect chemical ratio fidelity |

---

## 3. Dynamic Mid-Production Injection Analysis

### The Failure Mode in the Unmodified Baseline
In high-mix manufacturing, recipe changes occur dynamically.
* **Recipe 13** is a dye-only formulation ($0.0\text{ mL T100}$, $50.0\text{ mL T200}$, $20.0\text{ mL T400}$).
* Under fixed PLC sequence, all carrier recipes end by pumping T-100, leaving $V_{\text{tube}} = 8.2\text{ mL}$ of clear water in the nozzle.
* When Recipe 13 is dispensed in sequential order, the $8.2\text{ mL}$ of water is pushed straight into the vial:
  $$\text{Dilution} = \frac{8.2}{70.0 + 8.2} \approx 10.5\% \text{ to } 11.7\% \text{ concentration drop}$$
* In addition, recipes with target volume $\ge 75\text{ mL}$ (Recipe 9: $80\text{ mL}$, Recipe 10: $75\text{ mL}$) receive uncompensated liquid, reaching $83.2\text{--}88.2\text{ mL}$ and creating severe spill risks during MagneMover transit.

### The Solution: PCMC Option B + Backup Option A
1. **Initial Task (Option B):** 
   * Carrier-free recipes (Recipe 13) are grouped into a pure dye block at shift start with the nozzle dry/primed with dye.
   * A **Bridge Vial** (Recipe 7: Pale Blue, $50\text{ mL T400}$, $20\text{ mL T100}$) absorbs the trailing blue dye with hold-up credit, dispenses T-100 last, and transitions the line to clear diluent with **0.0 mL waste**.
2. **Halfway Arrival (Backup Option A):**
   * At Step 50, the line holds T-100 water when 10 new Recipe 13s arrive in List 2.
   * Option A executes a **single 8.2 mL boundary prime/purge** to the waste drain, evacuating water and priming the line with dye.
   * All 10 Recipe 13s are filled in pure dye mode with zero dilution defects.
   * An optimal **Return Bridge Vial** (Recipe 7: Pale Blue) is selected from the remaining carrier pool to absorb the resident dye, credit the volume, flush with T-100, and restore the line cleanly.
   * The remaining 139 carrier recipes execute with zero defects and zero waste.

---

## 4. Recipe Breakdown (Recipes 5–14)

| Recipe ID | Name | Target T100 | Target T200 | Target T400 | Total Vials | Baseline Defect Types | Baseline Defect Rate | PCMC Defect Rate | Waste Incurred |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **5** | Low Even | 10.0 mL | 10.0 mL | 10.0 mL | 19 | None (Clean) | 0% (0/19) | **0% (0/19)** | 0.0 mL |
| **6** | Pale Orange | 30.0 mL | 10.0 mL | 0.0 mL | 24 | Color Scrap (Blue dye carryover) | 8% (2/24) | **0% (0/24)** | 0.0 mL |
| **7** | Pale Blue | 20.0 mL | 0.0 mL | 50.0 mL | 18 | None (Bridge Transitions) | 0% (0/18) | **0% (0/18)** | 0.0 mL |
| **8** | Even Triple | 20.0 mL | 20.0 mL | 20.0 mL | 26 | None (Clean) | 0% (0/26) | **0% (0/26)** | 0.0 mL |
| **9** | Dark Saturate | 10.0 mL | 35.0 mL | 35.0 mL | 21 | Spill Risk (88.2 mL > 80 mL) | 100% (21/21) | **0% (0/21)** | 0.0 mL |
| **10** | Pale Wash | 45.0 mL | 15.0 mL | 15.0 mL | 21 | Spill Risk (83.2 mL > 80 mL) | 100% (21/21) | **0% (0/21)** | 0.0 mL |
| **11** | High Blue | 10.0 mL | 10.0 mL | 50.0 mL | 19 | None (Clean) | 0% (0/19) | **0% (0/19)** | 0.0 mL |
| **12** | High Orange | 10.0 mL | 50.0 mL | 10.0 mL | 16 | None (Clean) | 0% (0/16) | **0% (0/16)** | 0.0 mL |
| **13** | Orange+Blue | 0.0 mL | 50.0 mL | 20.0 mL | 18 | Dilution Defect (+8.2 mL Water) | 100% (18/18) | **0% (0/18)** | 8.2 mL (Option A Boundary) |
| **14** | Clear+Blue | 60.0 mL | 0.0 mL | 10.0 mL | 18 | None (Clean) | 0% (0/18) | **0% (0/18)** | 0.0 mL |

---

## 5. Isaac Sim Digital Twin Rendering Verification

* **Total Simulated Images:** 400 images (200 baseline + 200 optimized) + 200 side-by-side composite cards.
* **Storage Location:** [`mixing/test/`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/test/)
  * Baseline Renders: `mixing/test/baseline/vial_001_...png` to `vial_200_...png`
  * Optimized Renders: `mixing/test/optimized/vial_001_...png` to `vial_200_...png`
  * Comparison Cards: `mixing/test/comparison/comparison_vial_001.png` to `comparison_vial_200.png`
* **Reports:**
  * Full Markdown Report: [`mixing/test/README.md`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/test/README.md)
  * Interactive HTML Auditor: [`mixing/test/report.html`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/test/report.html)
