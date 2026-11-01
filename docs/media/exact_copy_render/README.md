# Exact-copy presentation render (kinematic, NOT physics)

Isaac Sim replay of the successful real-robot run `shared/runs/2026-10-30_16-21-27_physical_policy`
(1 missing + 1 crooked cap -> all 28 seated). Made on branch `exact_copy_render_run`
(commit `4bdd7fb`, `src/remediation/render_exact_copy_kinematic.py`); source frames and
`plan.json`/`timeline.json` are in `shared/runs/2026-11-01_exact_copy_render_full_v1/`.

**What is faked:** the real `trace.jsonl` waypoints are replayed through IK and the robot is posed
every frame; there is no perception and no contact physics. The spare cap is attached to the
fingers at the real jaw-close and set seated in G17 at jaw-open; the crooked cap G23 starts at a
50 deg lean and is blended upright as the cone nose descends. Waypoints are re-anchored ~2 cm onto
the sim tray, grasp/release heights adjusted, and vials G05/G28 hidden to match the real tray.
The wrist-camera images are not rendered (the robot still dwells at each wrist view).

| File | View | Timing |
|---|---|---|
| `exact_copy_tablecam_realistic.mp4` | TableCam 1920x1080 | real trace pauses (2:22) |
| `exact_copy_tablecam_realistic_clean.mp4` | TableCam | same, no caption |
| `exact_copy_tablecam_fast.mp4` | TableCam | pauses cut to 0.4 s (1:16) |
| `exact_copy_sidecam_*.mp4` | SideCam 1280x720 | same three variants |

Captioned videos read "Isaac Sim kinematic replay of real run 2026-10-30 (presentation render, not physics)".
Physics-validated results go in `docs/media/actually_done/` instead.
