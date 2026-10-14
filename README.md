# Rockwell Automation + NVIDIA Innovation Lab Fall 2026

A four-week physical AI hackathon with Rockwell Automation, NVIDIA, MSOE, and UWM's Connected Systems Institute (CSI).
- **Event Article:** <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
- **Full Guide:** See [instructions.md](instructions.md)

---

## 1. Initial Setup (Always Run First)

`data.zip` contains the digital twin and asset models (gitignored due to size).

1. Ensure `data.zip` is in the repository root (or on Rosie, data is pre-populated in `/data/ai_club/team_14_2026/data/`).
2. Run the repository setup:

```bash
chmod +x setup.sh
./setup.sh
```

`setup.sh` standardizes data access under `./shared/data/` across all platforms:
- **On Rosie:** Symlinks `shared -> /data/ai_club/team_14_2026` so all teammates share storage, assets, and Isaac Sim installations.
- **On GB10 / Brev / Local:** Extracts `data.zip` into `./shared/data/`.
- Bootstraps the FANUC robot arm simulation (generates procedural textures and runs unit tests).

---

## 2. Running on Rosie (HPC Cluster)

On Rosie, jobs run on Tesla T4 compute nodes via Slurm using the team's shared standalone **Isaac Sim 6.1.0** install located at `./shared/isaacsim_6.1`.

### Rendering with `./render.sh`
You can run `./render.sh` directly from the repo root with any arguments.
- **On Rosie (Slurm login node):** It automatically submits a background batch job via `sbatch`.
- **On GB10 / Brev (Direct shell):** It automatically runs directly using Isaac Sim.

Outputs are organized cleanly into timestamped folders (`runs/<YYYY-MM-DD_HH-MM-SS>_<target>/`) containing `render.png` (or `frame_*.png`) and `meta.json`.

```bash
# Render filled vial (Frame 686)
./render.sh --target vial --state filled

# Render empty vial (Frame 1)
./render.sh --target vial --state empty

# Render a multi-frame animation sequence (e.g. frames 100 to 500 every 20 frames)
./render.sh --target vial --state sequence --start-frame 100 --end-frame 500 --step 20

# Render CapSeal tray inspection camera
./render.sh --target cap_seal

# Render FANUC robotic arm views (side, table, or wrist)
./render.sh --target fanuc --camera side
./render.sh --target fanuc --camera table

# Build & cache persistent FANUC scene in shared/scenes/fanuc_scene.usd
./render.sh --target build_fanuc
```

### Monitoring Jobs & Viewing Outputs (Rosie)
```bash
# Check queue status
squeue --me

# Stream job logs
tail -f logs/slurm-<jobid>.out
tail -f logs/slurm-<jobid>.err
```
Render outputs are organized into `runs/<timestamp>_<target>/`.

---

## 3. Running on GB10 / Brev / Dedicated Cloud GPU

On dedicated GPU instances (e.g., NVIDIA GB10 / Grace Blackwell `aarch64` or Brev `x86_64`), you have direct interactive shell access without Slurm queues.

### A. Connecting to GB10 via SSH (Windows PowerShell)
To configure fast passwordless SSH access to the GB10 node from Windows PowerShell:

1. **Copy your SSH public key to the node (in PowerShell):**
   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh hackathon3@spark3.ai.msoe.edu "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ```
   *(Or if using Git Bash / WSL: `ssh-copy-id -i ~/.ssh/id_ed25519 hackathon3@spark3.ai.msoe.edu`)*

2. **Add the host alias to your local SSH config (`notepad $env:USERPROFILE\.ssh\config`):**
   ```sshconfig
   Host GB10
       HostName spark3.ai.msoe.edu
       User hackathon3
       IdentityFile ~/.ssh/id_ed25519
       ForwardAgent yes
   ```

3. **Connect anytime from PowerShell or Windows Terminal:**
   ```powershell
   ssh GB10
   ```

### B. Setup & Installation
Run `./setup.sh` — it will extract data assets, bootstrap robot textures, and automatically install Isaac Sim 6.1 into `./shared/isaacsim_6.1`:

```bash
./setup.sh
```
*(Or install Isaac Sim independently anytime with `./scripts/install_isaac_standalone.sh`)*

### C. Running Simulations Directly
On GB10 or Brev, run the exact same `./render.sh` command (it automatically executes directly):

```bash
# Render filled vial
./render.sh --target vial --state filled

# Render multi-frame filling sequence
./render.sh --target vial --state sequence --start-frame 1 --end-frame 686 --step 30

# Render FANUC robotic arm side perspective
./render.sh --target fanuc --camera side
```

### D. Transferring Rendered Outputs to Your Laptop
- **From GB10 (SCP using configured host alias):**
  ```bash
  scp -r GB10:~/innovation-lab-fall-2026/runs/<run_folder> ~/Downloads/
  ```
- **From Brev:**
  ```bash
  brev copy <instance-name>:/home/ubuntu/innovation-lab-fall-2026/runs/<run_folder> ~/Downloads/
  ```