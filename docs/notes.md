1. Detecting defects with vials, go back and move them as needed
2. Detecting defective recipes (maybe purge ingredients if possible)
3. Grabbing and sorting objects
	1. **Also fix cap if possible (or remove and replace with new cap)**
	2. Fill missing holes? (figure that out later, maybe have 10 good vials in scrap tray to use as replacements)
	3. Remove tray once done loading and remove defective ones to avoid collision

Get highest total throughput of non defective products, also increase amount of functional products
Make it as easy to apply to any manufacturing line
Slapping a camera and extra arm on the existing line should be enough
Do for as cheap as possible, low budget

---

### Implementation Progress (October 2026):
- **Architecture**: Implemented Two-Tier Solution:
  - **Tier 1 (Real-Time Edge)**: YOLOv11-cls + Golden Delta baseline ([`../scripts/aperture_inspector.py`](../scripts/aperture_inspector.py)) running in <15ms for all 30 slots at 100% recall.
  - **Tier 2 (Offline Data Engine / In-Context Reasoning)**: Cosmos Reason / SmolVLM auto-labeler ([`../scripts/vlm_auto_labeler.py`](../scripts/vlm_auto_labeler.py)) to label 1,181 real line images in `../shared/data/csi_vial_fill_dataset/cap_seal/images/`.
- **Detailed System Documentation**: See [`inspection_system_architecture.md`](inspection_system_architecture.md).
- **Aperture Coordinates**: Calibrated G01-G30 ([`../scripts/fixture_g_apertures.json`](../scripts/fixture_g_apertures.json)) & S01-S30 ([`../scripts/fixture_s_apertures.json`](../scripts/fixture_s_apertures.json)).
- **Synthetic Digital Twin Generator**: Procedurally renders defect variations ([`../scripts/generate_synthetic_crops.py`](../scripts/generate_synthetic_crops.py)).
- **Next Up**: Phase 3 Robotic Remediation integration (`../shared/data/fanuc_er4ia_sim`).

