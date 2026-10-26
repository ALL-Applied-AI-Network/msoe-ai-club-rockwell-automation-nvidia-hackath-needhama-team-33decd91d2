# Shared model registry

`shared/inspection_models/` is the team location for reviewed model files used
by multiple jobs or by the physical-client policy. It is separate from the
immutable competition reference directory `shared/data/`.

| Model file | Purpose | Source training artifact | SHA-256 | Current physical-policy default |
| --- | --- | --- | --- | --- |
| `shared/inspection_models/remediation_yolo_best.pt` | Detect `good_cap`, `crooked_cap`, `missing_cap`, and `table_cap` from TableCam RGB images. | `runs/archive/2026-10-22_black_silver_fixture_yolo/training/weights/best.pt` | `a190157c9c2b17cb03d6d323783f7320af68df42744e0adf42e84586f00b1894` | Yes |

Before replacing a shared model, retain the original training artifact in
`runs/archive/`, evaluate the replacement, calculate its SHA-256 checksum, and
update this table and the default path in
`PhysicalAI-Hackathon-2026/policy.py` together.
