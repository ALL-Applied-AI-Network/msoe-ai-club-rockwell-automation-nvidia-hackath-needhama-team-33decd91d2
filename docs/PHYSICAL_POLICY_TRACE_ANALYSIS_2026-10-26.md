# Physical-policy trace analysis — 2026-10-26

This analysis covers the real-cell policy traces preserved in `shared/runs/`.
It records what the policy observed and commanded; it does not establish that a
commanded position contacted the intended physical feature.

| Trace | TCP command records | Detection result | Depth result |
| --- | ---: | --- | --- |
| `2026-10-26_15-41-05_physical_policy` | 50 | 528 `crooked_cap`, 1 `table_cap` | No detection had calibrated depth. This run predated RGB-D fusion and used the old fixed tray-height projection. |
| `2026-10-26_15-54-42_physical_policy` | 235 | 1,689 `crooked_cap`, 4 `missing_cap`; no `good_cap` or `table_cap` | 1,601 detection records had calibrated depth; 92 did not. |
| `2026-10-26_16-11-43_physical_policy` | 52 | 948 `crooked_cap`; no other class | 920 detection records had calibrated depth; 28 did not. |

## Findings

The saved annotated images show that the synthetic-data YOLO model is calling
ordinary seated yellow caps `crooked_cap`. It also finds cap-like objects away
from the tray in the last trace. This is real-to-synthetic domain shift, not a
class-ID mapping problem: the checkpoint mapping is `good_cap`, `crooked_cap`,
`missing_cap`, and `table_cap` in that order.

The 15:41 run should not be used to judge position accuracy. It had no aligned
depth point, yet planned presses from a fixed simulated tray plane at base Z
0.269735 m. The later two traces did use RGB-D fusion. For example, the first
target in the 15:54 trace had a calibrated depth of 0.665 m and a fused base
point of `[0.4506, -0.0482, 0.1963]` m. The final press command was derived
from that fused point.

The trace cannot measure the observed "high and off to the side" error because
it lacks the physical point at which the custom fingers touched the cap. The
current policy explicitly uses a zero custom-tool offset in all three axes.
Therefore the final command sends the robot's nominal tool origin to the
camera-derived cap point, rather than sending the actual center of the custom
press face there. A real measured tool offset is required before correcting
that error numerically.

## Changes made from these traces

The contact controller now requires `depth_available: true` before it will
select any detection for a pickup, placement, or press. It will hold rather
than using the old plane-projected fallback. This prevents a depthless RGB box
from initiating physical contact, but it does not repair the YOLO class error.

## Next physical measurements

Use one upright, fully seated yellow cap in a clear tray location. With the
fingers vertical, jog the robot slowly until the intended center of the custom
press face is directly above the cap center, then record its robot TCP X and Y
values. Lower slowly until that press face first contacts the cap and record
the TCP Z value. Compare those three TCP coordinates with the RGB-D point
logged for the same cap. The signed difference is the custom-tool contact
offset needed for `cap_press_offset`; repeat the same procedure with the
grasp feature for pickup and the placement face for placement.

Before another autonomous full repair, label representative real TableCam
frames as `good_cap`, `crooked_cap`, `missing_cap`, and `table_cap`, including
the real lighting, robot shadow, tray, and loose yellow caps. Fine-tune and
evaluate the detector on a held-out set of those real images. The current
model cannot safely distinguish the required physical states.
