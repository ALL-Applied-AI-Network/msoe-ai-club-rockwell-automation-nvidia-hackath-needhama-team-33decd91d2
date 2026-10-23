# AGENTS.md — Developer & AI Agent Guidelines
## Rockwell Automation + NVIDIA Innovation Lab Fall 2026 (Team 14)
### Fluid Mixing Optimization & Digital Twin Simulation

---

## 1. Project Mission & Problem Statement

* **Context:** The CSI vial filling testbed utilizes a shared manifold line and single dispensing nozzle fed by four reagent tanks.
* **The Core Problem:** Residual liquid volume ($V_{\text{tube}} \approx 8.2\text{ mL}$) remains in the dispensing nozzle after each vial fill. When the recipe changes, this residual volume carries over into the next vial, causing chemical cross-contamination and off-color defects (e.g. orange residue turning a subsequent blue fill into an off-spec green).
* **The Solution (PCMC):** Predictive Carryover Minimization & Compensation framework:
  1. **Batch Reordering:** Sequentially orders recipes into a continuous Hamiltonian path where consecutive vials **always** share at least one compatible material.
  2. **Intra-Vial Sequencing:** Sequences the inherited resident fluid first and the handover bridging fluid last.
  3. **Volume Compensation:** Applies a hold-up credit on the entry fluid:
     $$V_{\text{pump}} = \max(0, V_{\text{target}} - V_{\text{tube}})$$
     delivering 100% accurate recipe volume with **zero dump-waste purges** and **zero cross-contamination**.
  4. **Digital Twin Validation:** Simulates and validates the scheduled batch in NVIDIA Isaac Sim using RTX PathTracing / RayTracedLighting.

---

## 2. Hard Operating Constraints & Factory Rules (MAIC Library)

All agents, scripts, and algorithms working in this repository **must strictly observe** the operating constraints from the [MAIC Fluid Mixing Specification](https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing):

| Constraint | Rule Specification | Rationale / Failure Mode |
| :--- | :--- | :--- |
| **Vial Volume** | Total volume per vial $\le 80.0\text{ mL}$ | Prevents liquid spillage during MagneMover cart transit. |
| **Component Min** | Each active component must be $\ge 10.0\text{ mL}$ ($0$ or $10\text{--}80\text{ mL}$) | Prevents dosing inaccuracy from low-volume pump strokes. |
| **Tank T-200 Limit** | No more than $50.0\text{ mL}$ of T-200 (Orange Dye) per recipe | Exceeding 50 mL faults the Rockwell PLC and requires a manual plant-manager reset. |
| **Tank T-300 Ban** | **Strictly 0.0 mL for T-300** | Tank T-300 is out of service and has a configuration issue that faults the line. |
| **Recipe Integrity** | Do not alter recipe formulations without plant-manager guidance | Target delivered chemical amounts must match original factory specs 1:1. |
| **PLC Dispense Order** | Physical PLC dispenses in fixed sequence: `T-200 -> T-400 -> T-100` | Skipping unused tanks. When present, **T-100 (clear diluent) is dispensed last**, leaving clear fluid in the tube. |
| **Queue Capacity** | Up to **3 recipes** can be stored simultaneously in the Batch Scheduler | Production campaigns are fed to the PLC in rolling 3-recipe blocks. |
| **Residual Carryover** | Leftover fluid from prior batches must be tracked and credited | Always initialize `BatchReorderer` with `initial_tube_fluid` and track `resident_fluid_passed`. |

> **Enforcement:** Programmatic validation is implemented in [`mixing/solution.py`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/solution.py) via `validate_operating_constraints()`. Violations raise a `ValueError` immediately upon instantiation.

---

## 3. Code Architecture & Object Model

The optimization pipeline is organized in the `mixing/` module:

```
mixing/
├── solution.py               # Solution class (data structures, optics, constraints)
├── batch_reorderer.py        # BatchReorderer class (Hamiltonian path, intra-vial sequencing, compensation)
├── driver.py                 # Driver pipeline (Recipe 5-14 initialization, scheduling, JSON export)
├── dispense_schedule.json    # Generated execution schedule consumed by Isaac Sim
├── isaac_dispense_sim.py     # Isaac Sim SimulationApp connector & RTX renderer
└── video_tube_estimator.py   # Computer vision analysis script for nozzle hold-up volume estimation
```

### Core Classes & Responsibilities

1. **`Solution`** ([`mixing/solution.py`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/solution.py)):
   * Represents a single vial recipe.
   * `materials`: Array of material names (e.g. `["T100", "T200", "T400"]`).
   * `amounts`: Array of corresponding volume amounts in mL.
   * `shared_materials(other)` / `is_compatible_with(other)`: Set intersection checking for bridging fluids.
   * `compute_rgb_color()`: Subtractive dye optical blending (Orange T200 + Blue T400 $\to$ Olive Green; T100 dilutes concentration).

2. **`BatchReorderer`** ([`mixing/batch_reorderer.py`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/batch_reorderer.py)):
   * `reorder_and_compensate(solutions)`:
     * Builds adjacency graph where edges represent shared reagents.
     * Computes a Hamiltonian path using depth-first search with heuristic edge weighting.
     * Chains intra-vial dispense order: `[entry_material, intermediate..., handover_material]`.
     * Calculates compensated pump amounts: $V_{\text{pump}} = \max(0, V_{\text{target}} - V_{\text{tube}})$.
     * Seamlessly inherits tube fluid from the previous batch (`initial_tube_fluid`) and records the trailing fluid (`resident_fluid_passed`).

3. **`mixing/driver.py`**:
   * Initializes benchmark Recipes 5–14 (from CSI Video 5).
   * Runs `BatchReorderer` with $V_{\text{tube}} = 8.2\text{ mL}$.
   * Emits [`mixing/dispense_schedule.json`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/mixing/dispense_schedule.json).

---

## 4. NVIDIA Isaac Sim Digital Twin Guidelines

### USD Stage & Prim Paths
* **Fill Station USD Stage:** `stations/fill_station/fill_station.usda`
* **Full Line Stage:** `csi_line.usda`
* **Camera Prim:** `/FillStation/FillStation_LocalRoot/FillCam/FillCam` (640 × 480 pinhole).
* **Liquid Mesh Prim:** `/FillStation/FillStation_LocalRoot/Fill_liquid/Fill_liquid`
* **Liquid Material Shader:** `/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF`
* **Jet Stream Material Shader:** `/FillStation/_materials/USD_jet_approximation/Principled_BSDF`

### RTX Rendering Rules
* **Do NOT use `UsdGeom.Gprim.GetDisplayColorAttr()` on `/Fill_liquid`:** In RTX PathTracing / RayTracedLighting, viewport display colors are ignored.
* **Always modify shader inputs directly:**
  ```python
  liquid_shader.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*rgb))
  liquid_shader.GetAttribute("inputs:opacity").Set(0.68)
  ```

### Mesh Base-Anchoring & Timecode Rules
The liquid mesh is pre-animated across frames 1–686. Scaling the mesh geometry from its origin changes its base height, causing the liquid column to float in mid-air above the vial holder if not anchored:
* **Keyframe Timecodes:**
  * $V \le 32.0\text{ mL} \to \text{tc} = 220.0$, $V_{\text{ref}} = 30.0\text{ mL}$
  * $V \le 45.0\text{ mL} \to \text{tc} = 301.0$, $V_{\text{ref}} = 40.0\text{ mL}$
  * $V \le 65.0\text{ mL} \to \text{tc} = 550.0$, $V_{\text{ref}} = 60.0\text{ mL}$
  * $V > 65.0\text{ mL} \to \text{tc} = 686.0$, $V_{\text{ref}} = 70.0\text{ mL}$
* **Scale Factor:** $s_z = V / V_{\text{ref}}$
* **Anchoring Offset:** In local coordinates, the mesh base is at $Z = 0.116\text{ in} \approx 0.00295\text{ m}$. To keep the bottom vertex pinned flush on the glass floor:
  $$T_z = (1.0 - s_z) \times 0.00295\text{ m}$$
  ```python
  trans_attr.Set(Gf.Vec3d(-0.000015, 0.0, tz))
  scale_attr.Set(Gf.Vec3f(1.0, 1.0, vol_scale))
  ```

---

## 5. HPC Execution & Testing (Rosie Cluster & Slurm)

* **Cluster Environment:** MSOE Rosie Supercomputer (Ubuntu 20.04, GLIBC 2.31).
* **CRITICAL CONSTRAINT:** **Never run Isaac Sim (`python.sh` / Kit) directly on the login node.** It will fail due to GLIBC 2.31 incompatibility.
* **Always submit rendering jobs via Slurm `sbatch`:**
  * Render wrapper: [`render.sh`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/render.sh)
  * Slurm script: [`scripts/submit_isaac_job.slurm`](file:///home/ad.msoe.edu/erdmannap/msoe-ai-club-rockwell-automation-nvidia-hackath-needhama-team-33decd91d2/scripts/submit_isaac_job.slurm)
  * Partition: `teaching` (GPU compute nodes, e.g. `dh-node2` with Tesla T4 / RTX GPUs).

### Standard Execution Commands

```bash
# 1. Run optimization driver and generate schedule:
python3 -m mixing.driver

# 2. Submit full 10-vial mixing simulation to Rosie GPU cluster:
./render.sh --target mixing

# 3. Or simulate/render a specific vial step (e.g. Step 2: Recipe 6 Pale Orange):
./render.sh --target mixing --vial-step 2

# 4. Monitor active Slurm jobs:
squeue -u $USER

# 5. Check job logs:
cat logs/slurm-<JOB_ID>.out
```

> **Sandbox Note:** `sbatch` connects to `slurmctld` over port 6817. When running Slurm commands from AI coding agent environments, run with `BypassSandbox: true` if standard sandbox isolation blocks cluster network sockets.

---

## 6. Success Metrics & Verification Checklist

When developing or evaluating changes, ensure the following benchmarks pass:

- [ ] **100% Transition Compatibility:** All consecutive pairs $(s_i, s_{i+1})$ must have $\ge 1$ shared active material.
- [ ] **Zero Dump Waste:** Purge dump volume must remain $0.0\text{ mL}$.
- [ ] **Zero Cross-Contamination:** Delivered chemical amounts inside each vial match target amounts with 100% fidelity.
- [ ] **Hold-Up Savings:** Reagent credit applied equals $V_{\text{tube}} \times (\text{Number of transitions} + 1)$.
- [ ] **Spill Prevention:** All vials $\le 80.0\text{ mL}$.
- [ ] **Equipment Safety:** Tank T-200 $\le 50.0\text{ mL}$, Tank T-300 strictly $0.0\text{ mL}$.
- [ ] **Visual Accuracy:** Fluid columns in Isaac Sim renders rest flush on the vial carrier base without gaps or clipping.

