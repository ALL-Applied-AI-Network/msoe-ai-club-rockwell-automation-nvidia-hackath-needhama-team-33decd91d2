# Rockwell Automation + NVIDIA Innovation Lab Fall 2026

A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute (CSI). Teams develop solutions for one of three challenges from a real factory testbed:
1. **Dataset 1: Final Vial Inspection** – Missing/crooked caps, damaged vials, empty fixtures.
2. **Dataset 2: Fluid Mixing Station 3** – Synchronized camera views to detect fill and color anomalies.
3. **Dataset 3: Grasping and Sorting** – Standalone FANUC arm sorting and digital twin.

Full event documentation and guide:
- **Event Article:** <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- **Local Documentation:** See [instructions.md](instructions.md)

---

## Setup & Data

The repository includes a setup script to unpack the dataset archive into the `.gitignore`d `data/` directory:

```bash
./setup.sh
```

This extracts `data.zip` into `data/`, providing:
- `data/csi_digital_twin_v1/`
- `data/csi_vial_fill_dataset/`
- `data/fanuc_er4ia_sim/`