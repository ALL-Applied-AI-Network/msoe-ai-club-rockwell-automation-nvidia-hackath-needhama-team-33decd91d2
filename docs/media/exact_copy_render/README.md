# Staged Kinematic Replay of the Real 2026-10-30 Run (NOT physics)

**These videos are presentation renders, not simulation results.** They show the Isaac Sim digital twin
replaying the successful real-robot run `shared/runs/2026-10-30_16-21-27_physical_policy/`. The robot
moves are scripted and the cap poses are written directly. The robot does not use physics contact or
perception at any point.

- Generated on branch `exact_copy_render_run` (commit `4bdd7fb`) by
  `src/remediation/render_exact_copy_kinematic.py`. That script exists only on that branch.
- Source run: `shared/runs/2026-11-01_exact_copy_render_full_v1/`. That folder holds the PNG frames,
  `plan.json` and `timeline.json`.

## What is faked

- **Robot motion.** The real `trace.jsonl` TCP actions are replayed through the repo IK, and the joints
  are set directly every frame. Gravity and collisions are off for the caps, tray and robot.
- **Position offsets.** Waypoints are re-anchored about 2 cm onto the sim's spare cap, missing vial
  (G17) and crooked vial (G23). This makes up for the mismatch between the real calibration and the sim.
- **Spare cap.** The cap is attached to the fingers when the jaws close. When they open, it is set
  seated in G17.
- **Crooked cap.** G23 starts at a 50° lean, like the real one. It is blended upright while the cone
  nose comes down. The real stroke heights are kept: +50 mm and +20 mm, down to 3 mm below the rim.
- **Hidden vials.** G05 and G28 are hidden so the tray layout matches the real tray. The run starts
  with 1 missing, 1 crooked and 26 good caps, and ends with all 28 seated.
- **Wrist camera.** Its views are not rendered. The robot still visits each wrist-camera pose with the
  real dwell times.

## Video catalog

| File | Timing | Viewpoint | Duration | Caption |
|---|---|---|---|---|
| `exact_copy_tablecam_realistic.mp4` | Real trace timing | TableCam 1920x1080 | 2:23 | Yes |
| `exact_copy_tablecam_realistic_clean.mp4` | Real trace timing | TableCam 1920x1080 | 2:23 | No |
| `exact_copy_tablecam_fast.mp4` | Pauses cut to 0.4 s | TableCam 1920x1080 | 1:16 | Yes |
| `exact_copy_sidecam_realistic.mp4` | Real trace timing | SideCam 1280x720 | 2:23 | Yes |
| `exact_copy_sidecam_realistic_clean.mp4` | Real trace timing | SideCam 1280x720 | 2:23 | No |
| `exact_copy_sidecam_fast.mp4` | Pauses cut to 0.4 s | SideCam 1280x720 | 1:16 | Yes |

The captioned videos read: "Isaac Sim kinematic replay of real run 2026-10-30 (presentation render,
not physics)".

Known limitation: in the TableCam view, the gripper body hides the moment the spare cap is released.
