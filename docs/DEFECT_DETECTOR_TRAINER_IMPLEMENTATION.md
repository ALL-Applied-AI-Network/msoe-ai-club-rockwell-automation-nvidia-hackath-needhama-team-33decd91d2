# Defect Detector Trainer Implementation Tracker

## Purpose

This document is the persistent implementation checklist for the generalizable defect detector trainer described in `docs/plan.md`. Update it after each implementation session so the next contributor can see what is complete, what was validated, and what should happen next.

The trainer builds a real-image YOLO dataset from a small human-reviewed seed set, class-agnostic LocateAnything proposals, and VLM-assisted crop labels. It may train from a standard YOLO checkpoint or fine-tune a synthetic checkpoint.

This work extends dataset generation and inspection tooling. It must not introduce an alternative remediation runner or change the compliant, collision-aware remediation contract.

## Fixed Design Decisions

- LocateAnything localizes objects; it does not assign defect classes.
- The VLM classifies padded object crops by visual appearance.
- YOLO labels use the original, unpadded proposal boxes.
- Crop width and height are multiplied by `1.25`, adding 12.5 percent of the original size on each side before boundary clipping.
- Dataset splits are assigned before VLM labeling and grouped by capture sequence, tray state, specimen, or session.
- VLM reference crops come only from the training split.
- The real test split is fully human-reviewed and is never used for prompt or threshold selection.
- The vial-cap appearance classes are `good_cap`, `crooked_cap`, and `missing_cap`.
- Loose intact caps are labeled `good_cap`, not `table_cap`.
- Runtime depth Z distinguishes a seated `good_cap` from a spare cap on the table.
- Vial-cap localization permits at most 35 merged object proposals per image; over-limit images require review and are never silently truncated.
- Invalid or ambiguous depth never produces an autonomous spare-cap observation.
- Existing four-class datasets and checkpoints remain unchanged for reproducibility.
- GPU VLM inference and YOLO training run through Slurm.
- Generated artifacts go under `shared/runs/YYYY-MM-DD_<project>_weak_labeling/`; `shared/data/` remains read-only.

## Target Repository Layout

```text
src/inspection/defect_training/
    __init__.py
    config.py
    schemas.py
    boxes.py
    seed_import.py
    locate_anything.py
    crop_generation.py
    vlm_classification.py
    label_review.py
    yolo_export.py
    dataset_validation.py
    depth_qualification.py

scripts/datasets/
    import_defect_seeds.py
    generate_defect_proposals.py
    generate_defect_crops.py
    classify_defect_crops.py
    finalize_real_defect_dataset.py
    convert_remediation_dataset_to_three_classes.py
    build_mixed_defect_dataset.py

scripts/training/
    train_defect_detector.py
    evaluate_defect_detector.py

scripts/vlm/
    generate_defect_proposals.slurm
    classify_defect_crops.slurm

scripts/training/
    train_defect_detector.slurm

tests/
    test_defect_training_boxes.py
    test_defect_training_schemas.py
    test_defect_training_splits.py
    test_defect_training_export.py
    test_depth_qualification.py
```

Names may be adjusted during implementation if existing repository conventions provide a better home. Record any change in the decision log.

## Canonical Run Layout

```text
shared/runs/YYYY-MM-DD_<project>_weak_labeling/
    config.yaml
    source_manifest.json
    seeds/
        manifest.jsonl
        crops/
        contact_sheets/
    proposals/
        records/
        overlays/
        manifest.json
    crops/
        images/
        manifest.jsonl
    classifications/
        records/
        manifest.jsonl
    review/
        queue.jsonl
        contact_sheets/
        corrections.jsonl
    dataset/
        images/train/
        images/val/
        images/test/
        labels/train/
        labels/val/
        labels/test/
        previews/
        data.yaml
        manifest.json
        build_report.json
    training/
```

Do not create loose debug artifacts in `shared/runs/`.

## Canonical Object Schema

Every detected or manually labeled object has a stable record:

```json
{
  "schema_version": 1,
  "object_id": "capture04_frame0042__0007",
  "image": "images/capture04_frame0042.png",
  "image_size": [1920, 1080],
  "split": "train",
  "split_group": "capture04",
  "source_box_xyxy": [412.0, 283.0, 493.0, 361.0],
  "crop_box_xyxy": [402.0, 273.0, 503.0, 371.0],
  "localization_source": "locateanything",
  "localization_prompt": "cap or open top of each vial",
  "appearance_label": "good_cap",
  "label_source": "vlm",
  "confidence": 0.94,
  "status": "accepted",
  "reference_ids": ["seed_good_001", "seed_good_004"]
}
```

Required invariants:

- Pixel boxes use `[x1, y1, x2, y2]` with an exclusive lower-right boundary.
- `source_box_xyxy` remains unchanged after localization.
- `crop_box_xyxy` is `null` immediately after localization, then is derived deterministically from the source box and crop configuration during crop generation.
- An effective training label is accepted only when `status` is `accepted` or `human_corrected`.
- Raw model responses and human corrections remain available in provenance records.

## Implementation Checklist

### Milestone 0 — Establish a Reviewed Pilot Set

- [x] Select a small pipeline-validation set covering seated, crooked, and missing caps across three real capture sequences.
- [x] Define the grouping key for the available real images.
- [x] Assign pilot train, validation, and test groups before labeling.
- [ ] Manually label all objects in the pilot test set.
- [x] Identify reviewed training examples providing several references per class.
- [ ] Confirm loose intact table caps use `good_cap`.
- [x] Record the pilot source and grouping policy in a source manifest.

Acceptance criteria:

- Every pilot image belongs to exactly one split group.
- No split group crosses train, validation, and test.
- The pilot test set is fully reviewed and contains representative examples of every class being scored.

### Milestone 1 — Configuration, Schemas, and Box Utilities

- [x] Add a versioned YAML project configuration schema.
- [x] Validate class names, unique IDs, crop scale, thresholds, and split fractions.
- [x] Add typed canonical object records for proposal and classification state.
- [x] Implement YOLO-to-pixel and pixel-to-YOLO conversions.
- [x] Implement centered 1.25-times box expansion with image-boundary clipping.
- [x] Implement stable object IDs independent of processing order.
- [x] Add JSONL readers and atomic record writers with schema-version checks.
- [x] Add tests for box expansion, clipping, degenerate boxes, and coordinate round trips.

Acceptance criteria:

- Unit tests prove exact crop behavior for centered, edge, corner, one-pixel, and invalid boxes.
- Source geometry is immutable after record creation.
- Configuration errors fail before any dataset artifacts are written.

### Milestone 2 — Seed Annotation Import

- [x] Accept a standard YOLO seed dataset and project configuration.
- [x] Verify that images, labels, and class IDs agree with the configuration.
- [x] Convert seed labels into canonical object records.
- [x] Generate 1.25-times reference crops.
- [x] Generate per-class contact sheets and full-image overlays.
- [x] Reject seeds outside the training split from the VLM reference pool.
- [x] Write a seed manifest containing provenance and class counts.

Acceptance criteria:

- Every imported box appears in both a canonical record and an audit overlay.
- Reference crops are sourced only from training groups.
- Importing the same source twice produces identical object IDs and crop geometry.

### Milestone 3 — Generic LocateAnything Proposals

- [x] Reuse the existing LocateAnything worker behind a lightweight, testable proposal module.
- [x] Accept an unlabeled image directory, configuration, split manifest, and output directory.
- [x] Use one configurable object-of-interest prompt by default.
- [x] Preserve raw responses, model identity, adapter identity, seed, and inference settings.
- [x] Clamp coordinates and reject degenerate predictions.
- [x] Apply configurable NMS or proposal clustering.
- [x] Resume at image granularity.
- [x] Record images with zero proposals.
- [x] Flag unusual proposal counts without assuming a fixed aperture grid.
- [x] Generate full-image proposal overlays.
- [x] Add a Slurm launcher for full proposal generation.

Acceptance criteria:

- Proposal recall is measured on the fully reviewed pilot set.
- Interrupted inference resumes without duplicating records.
- No class label is inferred during localization.
- Raw responses allow every parsed box to be audited.

### Milestone 4 — Candidate Crop Generation

- [ ] Generate deterministic 1.25-times crops for every accepted proposal.
- [ ] Retain source and crop coordinates in the crop manifest.
- [ ] Record boundary clipping and crop dimensions.
- [ ] Refuse duplicate object IDs with conflicting geometry.
- [ ] Generate an overview contact sheet for the pilot run.
- [ ] Add tests proving that crop generation does not modify the source box.

Acceptance criteria:

- Regenerating crops from the same manifest produces identical geometry.
- Every crop maps back to exactly one source image and proposal.
- Clipped and anomalously small crops are flagged for review.

### Milestone 5 — Few-Shot VLM Classification

- [ ] Define a backend-neutral VLM classifier interface.
- [ ] Construct prompts from configuration class descriptions and reviewed reference crops.
- [ ] Require a structured label, confidence, and short reason.
- [ ] Support the explicit `uncertain` result.
- [ ] Reject unknown labels and malformed responses.
- [ ] Implement direct multiclass classification.
- [ ] Implement a second reference-similarity or equivalent consensus pass.
- [ ] Apply configurable acceptance and review thresholds.
- [ ] Preserve model, prompt, references, raw response, and parsed result.
- [ ] Resume at object granularity.
- [ ] Add a Slurm launcher for batch crop classification.

Initial decision policy:

```text
confidence >= 0.90 and passes agree -> accepted
0.70 <= confidence < 0.90          -> review
confidence < 0.70                  -> rejected pending review
disagreement, uncertain, invalid   -> review or reject
```

Acceptance criteria:

- Accuracy, confusion matrix, abstention rate, and confidence calibration are measured on reviewed pilot crops.
- Invalid VLM output cannot enter the YOLO dataset.
- Test images are never included as reference examples.

### Milestone 6 — Review Queue and Corrections

- [ ] Build a review queue for low confidence, disagreement, uncertainty, clipping, size anomalies, and count anomalies.
- [ ] Include a configurable random sample of high-confidence accepted labels.
- [ ] Generate crop and full-image review sheets.
- [ ] Define a simple correction JSONL format.
- [ ] Apply corrections deterministically without erasing original predictions.
- [ ] Report human correction rate by class and confidence band.

Acceptance criteria:

- Every non-accepted object has an explicit reason.
- A correction changes only the effective label/status and preserves provenance.
- Reapplying the same corrections is idempotent.

### Milestone 7 — YOLO Dataset Finalization

- [ ] Join accepted appearance labels to original source boxes.
- [ ] Export train, validation, and test image/label directories.
- [ ] Write empty label files for reviewed negative images.
- [ ] Generate `data.yaml` from configured class order.
- [ ] Generate final full-image overlays.
- [ ] Write a build report with image, object, class, source, confidence, and review counts.
- [ ] Write a versioned dataset manifest and hashes for inputs and configuration.
- [ ] Reject split leakage, missing files, invalid coordinates, and unknown classes.
- [ ] Add end-to-end export tests.

Acceptance criteria:

- Every YOLO row traces back to one canonical object record.
- Every exported class ID agrees with `data.yaml`.
- Padded crop geometry never appears in YOLO labels.
- The validation and test splits pass their required human-review policy.

### Milestone 8 — Three-Class Synthetic Conversion

- [ ] Add a non-destructive converter for existing four-class synthetic datasets.
- [ ] Map `good_cap` and `table_cap` to the new `good_cap` ID.
- [ ] Preserve `crooked_cap` and `missing_cap`.
- [ ] Regenerate `data.yaml` and dataset manifests.
- [ ] Produce before/after class counts and sample overlays.
- [ ] Verify that source datasets and checkpoints are untouched.

Acceptance criteria:

- Converted labels contain only the configured three classes.
- Box coordinates are byte-for-byte equivalent apart from class ID changes where possible.
- Conversion is deterministic and fully documented in the output manifest.

### Milestone 9 — Training and Evaluation

- [ ] Implement training from a standard YOLO checkpoint.
- [ ] Implement fine-tuning from a synthetic checkpoint.
- [ ] Implement reproducible real/synthetic replay composition.
- [ ] Avoid opaque bulk file duplication where a manifest or supported sampling strategy is sufficient.
- [ ] Record checkpoint, configuration, dataset hashes, hyperparameters, and environment.
- [ ] Evaluate on the fully reviewed real holdout.
- [ ] Evaluate synthetic regression when using a synthetic checkpoint or replay.
- [ ] Report per-class precision, recall, AP, and confusion matrices.
- [ ] Add Slurm training and evaluation launchers.

Acceptance criteria:

- Real holdout results are reported independently from synthetic results.
- The selected checkpoint improves the agreed real-data metric without unacceptable synthetic regression.
- No checkpoint is marked deployment-ready based only on weak labels or synthetic validation.

### Milestone 10 — Runtime Z Qualification

- [ ] Add robust depth sampling inside an eroded `good_cap` detection box.
- [ ] Ignore invalid depth and require a configurable minimum valid-pixel count.
- [ ] Use a robust median or validated surface percentile rather than one center pixel.
- [ ] Calculate expected seated and spare top heights from calibrated scene geometry.
- [ ] Calibrate seated and table tolerances from reviewed RGB-D examples.
- [ ] Emit `seated_good_cap`, `spare_cap`, or `ambiguous` operational roles.
- [ ] Map `spare_cap` to the existing controller's `table_cap` observation during migration.
- [ ] Ensure ambiguous or depthless detections cannot be selected as spares.
- [ ] Add unit tests and an end-to-end dry-run evaluation.

Expected reference heights:

```python
seated_z = rim_z + SPEC.flange_thickness
spare_z = surface_z + SPEC.plug_depth + SPEC.flange_thickness
```

Acceptance criteria:

- Reviewed evaluation reports seated/table accuracy and ambiguous rate.
- The operational release set contains zero false-spare selections.
- Existing collision-aware remediation behavior remains unchanged after the compatibility mapping.

### Milestone 11 — Active Learning Loop

- [ ] Run the trained YOLO detector over additional real captures.
- [ ] Rank uncertain, conflicting, rare, and visually novel examples.
- [ ] Review the highest-value examples rather than uniformly relabeling everything.
- [ ] Add reviewed corrections to a new version of the seed dataset.
- [ ] Rebuild and retrain with immutable input manifests.
- [ ] Track VLM abstention, correction rate, and detector performance across versions.

Acceptance criteria:

- Each iteration is reproducible from recorded manifests.
- New data improves held-out real performance or addresses a documented failure mode.

## Stage Evaluation Metrics

### Localization

- Proposal recall on reviewed objects.
- IoU and center error.
- Missed-object and duplicate-proposal rates.
- Zero-proposal image count.

### VLM Classification

- Accuracy and confusion matrix.
- Per-class precision and recall.
- Abstention and human-correction rates.
- Accuracy by confidence band.

### YOLO Detector

- Real-image precision, recall, mAP50, and mAP50-95.
- Per-class false positives and false negatives.
- Performance by capture session, lighting condition, and defect severity.
- Synthetic holdout regression where applicable.

### Z Qualification

- Seated-versus-table accuracy.
- Missing/invalid/ambiguous depth rate.
- False-spare and missed-spare rates.
- End-to-end spare selection success.

## Current Status

| Milestone | Status | Evidence / Output |
| --- | --- | --- |
| 0. Reviewed pilot set | In progress | Five-frame/three-sequence validation pilot assigned; fully human-boxed holdout and loose-cap confirmation remain |
| 1. Configuration and schemas | Complete | `src/inspection/defect_training/{config,schemas,boxes}.py`; 13 focused tests pass |
| 2. Seed import | Complete | `seed_import.py` and `import_defect_seeds.py`; pilot imported 105 objects and produced training-only references |
| 3. LocateAnything proposals | In progress | Jobs 290616/290617 completed; loose caps 2/2 localized, revised tray policy failed at 13/75 train+val recall |
| 4. Crop generation | Not started | — |
| 5. VLM classification | Not started | Existing cap-specific VLM experiments available for reference |
| 6. Review and corrections | Not started | — |
| 7. YOLO finalization | Not started | Existing fixture-specific exporters available for reference |
| 8. Synthetic conversion | Not started | Existing four-class datasets remain authoritative inputs |
| 9. Training and evaluation | Not started | Existing fine-tuning scripts available for refactoring |
| 10. Z qualification | Not started | Existing expected-height checks available for reuse |
| 11. Active learning | Not started | — |

## Next Action

Start with Milestone 0 and Milestone 1:

1. Identify the real-image directories and a reliable capture-group key.
2. Select and fully review a small pilot set.
3. Agree on the pilot split before using any image as a VLM reference.
4. Implement configuration validation, canonical schemas, stable IDs, and box utilities.
5. Run their unit tests before adding model inference.

Do not start a full LocateAnything or VLM batch until proposal recall and crop/classification behavior have been measured on the pilot set.

## Decision Log

Add dated entries whenever an implementation choice changes the fixed design or affects reproducibility.

| Date | Decision | Reason | Consequence |
| --- | --- | --- | --- |
| 2026-10-28 | Use three learned vial-cap appearance classes. | Loose and seated intact caps share visual identity; placement is geometric. | Convert old `table_cap` labels to `good_cap` in new derived datasets. |
| 2026-10-28 | Use Z only for seated-versus-table qualification. | The two states differ by approximately the 203.2 mm vial height, and aperture/vial XY references are unavailable. | Require calibrated height bands and an explicit ambiguous state. |
| 2026-10-28 | Keep the source proposal box separate from the padded VLM crop. | Context padding should not inflate YOLO ground truth. | Store and validate both boxes in every canonical record. |
| 2026-10-28 | Set the vial-cap proposal sanity limit to 35 objects per image. | The physical scene should not contain more than approximately 35 caps/openings. | Preserve all proposals for audit, flag the image, and block automatic acceptance rather than truncating silently. |

## Session Update Template

At the end of an implementation session, update the status table and append a short entry here:

```text
### YYYY-MM-DD — Short session title

Completed:
- ...

Validated:
- Command/test and result

Decisions:
- ...

Next:
- ...

Blocked by:
- None, or a concrete blocker requiring user input
```

### 2026-10-28 — Configuration, records, and box geometry

Completed:
- Added validated, versioned YAML project configuration dataclasses.
- Added immutable pixel boxes, exact 1.25-times expansion, clipping, and YOLO conversions.
- Added canonical object records, deterministic IDs, and atomic JSONL manifests.
- Added focused configuration, schema, and geometry tests.

Validated:
- `.conda_env/bin/python -m unittest tests.test_defect_training_boxes tests.test_defect_training_schemas` — 13 tests passed.
- `git diff --check` — passed.

Decisions:
- Stable IDs normalize coordinates to floats so integer and floating-point representations of identical geometry produce the same ID.
- Boundary clipping does not shift the opposite crop edge to preserve the requested size.

Next:
- Select and group the reviewed pilot images.
- Implement the YOLO seed importer and reference-crop audit outputs.

Blocked by:
- Pilot image source and grouping policy have not yet been selected.

### 2026-10-28 — Reviewed seed importer and validation pilot

Completed:
- Added the vial-cap trainer configuration and a leakage-safe three-sequence pilot split.
- Added generic YOLO seed import, split-manifest validation, training-only crop generation, overlays, contact sheets, and build reports.
- Imported the five existing reviewed TableCam frames as a non-destructive `/tmp` validation run.

Validated:
- `.conda_env/bin/python -m unittest tests.test_defect_training_boxes tests.test_defect_training_schemas tests.test_defect_training_seed_import` — 17 tests passed.
- Pilot import — 5 images and 105 objects: 43 `good_cap`, 50 `crooked_cap`, and 12 `missing_cap`.
- Split isolation — 60 training, 15 validation, and 30 test objects; related `run_155442` frames remain together.
- Reference isolation — only the 60 training objects generated VLM reference crops.
- Visual inspection — the generated `good_cap` contact sheet has centered crops with the intended context.

Decisions:
- Use the filename `run_HHMMSS` prefix as the grouping key for this pilot.
- Treat the current five-frame set as pipeline validation because classes/counts were human-reviewed but some box geometry was VLM-refined.
- Require a separately confirmed, fully human-boxed holdout before release evaluation.

Next:
- Add class-agnostic LocateAnything proposal generation against this validation pilot.
- Explicitly convert the two reviewed loose-cap examples from legacy `table_cap` to `good_cap` before adding them to training references.

Blocked by:
- The final release holdout still needs fully human-reviewed box geometry.

### 2026-10-28 — Class-agnostic LocateAnything proposals

Completed:
- Added a lightweight proposal module with response parsing, coordinate scaling, duplicate suppression, overlays, reports, canonical records, and image-level resume.
- Added a GPU command and Slurm launcher that load the existing LocateAnything worker only in the model process.
- Made crop geometry optional in localization-stage records; Milestone 4 will populate it deterministically.
- Submitted the five-image validation pilot as Slurm job `290615` using the base LocateAnything model.

Validated:
- `.conda_env/bin/python -m unittest tests.test_defect_training_boxes tests.test_defect_training_schemas tests.test_defect_training_seed_import tests.test_defect_training_locate_anything` — 20 tests passed.
- Resume tests prove completed images do not invoke the model again.
- Fingerprint tests reject resume when the model, adapter, prompt, or NMS policy changes.
- Slurm job `290615` entered `RUNNING` on `dh-dgxh100-1`.

Decisions:
- Keep GPU/model imports out of the reusable proposal module so geometry and resume behavior can be tested in the standard environment.
- Use a configuration fingerprint on every per-image record to prevent mixed inference settings in one run.
- Prefer the tighter box when duplicate generative outputs overlap above the configured NMS threshold.

Next:
- Inspect job `290615`, calculate proposal recall against the reviewed pilot, and tune the object prompt only from train/validation data.
- Implement deterministic candidate crop generation after the pilot proposal format is confirmed.

Blocked by:
- Milestone 3 acceptance remains pending until the GPU pilot completes and proposal recall is measured.

### 2026-10-28 — Multi-prompt vial-top and loose-cap localization

Completed:
- Generalized localization configuration from one prompt to an ordered set of object prompts.
- Added cross-prompt duplicate merging while retaining every prompt that supported a proposal.
- Changed the vial-cap configuration to localize both vial tops and loose caps on the work surface.
- Added leakage-safe train/validation assignments for the two reviewed loose-cap still images.
- Submitted revised tray job `290616` and loose-cap job `290617` to distinct output directories.

Validated:
- `.conda_env/bin/python -m unittest tests.test_defect_training_boxes tests.test_defect_training_schemas tests.test_defect_training_seed_import tests.test_defect_training_locate_anything` — 24 tests passed.
- Source compilation and `git diff --check` passed.
- Job `290616` entered `RUNNING`; job `290617` entered `PENDING` with a normal priority wait.

Decisions:
- Use `top of every clear plastic vial in the metal tray, whether capped or uncapped` for tray object proposals.
- Use `loose cap lying on the table or work surface` for spare-cap proposals.
- Keep both result streams appearance-agnostic; downstream VLM labeling assigns `good_cap`, `crooked_cap`, or `missing_cap`.
- Store prompt provenance per proposal so false positives and prompt behavior remain auditable.

Next:
- Measure tray proposal recall using train/validation only and verify both loose-cap boxes are localized.
- If localization passes, implement Milestone 4 candidate crop generation across the merged proposal manifests.

Blocked by:
- Waiting for Slurm jobs `290616` and `290617` to finish.

### 2026-10-28 — Multi-prompt pilot results

Completed:
- Inspected completed tray job `290616` and loose-cap job `290617`.
- Separated proposal counts by source prompt and measured class-agnostic train/validation recall.
- Visually audited a loose-cap overlay and a failed tray overlay.

Validated:
- Both Slurm jobs completed with exit code `0:0`.
- Loose-cap prompt localized exactly one reviewed loose cap in each of the two still images: 2/2 detected.
- Revised tray prompts matched 13 of 75 train/validation objects at IoU 0.50, or 17.3 percent recall.
- The vial-top prompt returned zero boxes in every image.
- The loose-cap prompt produced all tray-run proposals; on training frames these were small false positives on table markings rather than vial objects.

Decisions:
- Keep the loose-cap prompt for images containing the work surface; its two reviewed examples passed.
- Reject the broad vial-top prompt from the localization policy.
- Restore the original seated/crooked-cap prompt, which previously found 93/93 visible caps, and add a separate focused uncapped-vial-opening prompt.
- Do not begin candidate crop generation until the combined visible-cap and opening prompts meet the localization recall gate.

Next:
- Run prompt development only on train/validation images using the original visible-cap prompt plus a focused `open top of an uncapped clear plastic vial` prompt.
- Measure per-class recall and visually inspect false positives before locking the localization configuration.

Blocked by:
- Missing-cap localization is not yet qualified; the current broad vial-top prompt has zero recall.

### 2026-10-28 — Corrected cap and uncapped-opening prompt pilot

Completed:
- Restored the proven `cap or open top of each vial` prompt for visible seated and crooked caps.
- Added the focused `open top of an uncapped clear plastic vial in the metal tray` prompt.
- Retained the independently validated loose-cap prompt.
- Added train/validation-only filtering to prevent prompt development from touching the locked test group.
- Submitted corrected localization job `290625` to `proposals_v3_train_val`.

Validated:
- The expanded focused suite passes 27 tests.
- Source compilation and `git diff --check` pass.
- Split-filter testing proves test images are not passed to the inference backend.
- Slurm job `290625` entered the queue with a normal priority wait.

Decisions:
- Treat the 35-object limit only as an anomaly/review gate, never as a localization repair or truncation policy.
- Evaluate prompt changes on train/validation only; run the locked test split only after the localization policy is selected.

Next:
- On completion of job `290625`, measure visible-cap and missing-opening recall separately and audit the prompt-specific overlays.
- Lock the prompt policy only if localization recall meets the agreed gate; otherwise refine the opening prompt without using test data.

Blocked by:
- Waiting for Slurm job `290625` and missing-opening recall results.

### 2026-10-28 — Focused uncapped-opening prompt results

Completed:
- Inspected completed job `290625` and measured merged and per-prompt training recall.
- Visually audited the newest merged proposal overlay.
- Identified a Slurm export issue that reduced `train,val` to `train`; no locked test images were processed.

Validated:
- Job `290625` completed with exit code `0:0`.
- The original cap prompt matched all 52 visible good/crooked caps in the three training frames.
- The focused uncapped-opening prompt matched 51 visible caps but 0 of 8 missing-cap openings; it duplicated cap geometry rather than localizing openings.
- The loose-cap prompt matched 0 tray objects and added 16 false positives on table markings.
- Merged recall was 52/60, or 86.7 percent, with all eight misses belonging to `missing_cap`.
- One image produced 39 merged proposals and was correctly flagged by the 35-object anomaly gate; no proposals were truncated.

Decisions:
- Do not continue treating a missing cap as a directly localizable object; LocateAnything consistently localizes visible caps and not their absence.
- Evaluate localizing the containing vial as the object of interest, then classify a deterministic vial-top crop as good, crooked, or missing.
- Keep loose-cap localization as a separate visible-object stream.

Next:
- Pilot a whole-vial LocateAnything prompt on train/validation images.
- From each vial box, derive and visualize a consistent top-region crop for VLM condition classification.
- Confirm that vial localization covers every reviewed missing-cap vial before implementing general crop generation.

Blocked by:
- The original direct-opening design cannot meet missing-cap recall; the localization unit must change from cap/opening to vial or another visible containing object.

### 2026-10-28 — Whole-vial localization experiment

Completed:
- Added an isolated experiment configuration using `clear plastic vial in the metal tray` as the localization target.
- Fixed Slurm split filtering to accept `train+val` without the comma being interpreted as another exported variable.
- Made the Slurm wrapper accept an explicit project configuration.
- Submitted train/validation-only whole-vial job `290674` to `vial_proposals_train_val`.

Validated:
- The focused suite still passes 27 tests.
- Source compilation and `git diff --check` pass.
- Job `290674` entered `RUNNING` on `dh-dgxh100-1`.

Decisions:
- Keep the vial-localization experiment separate from the main configuration until coverage is measured.
- Evaluate vial proposals by containment of reviewed cap/opening centers rather than cap-box IoU, since the proposal geometry intentionally covers the containing vial.

Next:
- Measure vial coverage separately for good, crooked, and missing-cap examples.
- Visualize candidate top regions derived from detected vial boxes.
- Adopt the approach only if every missing-cap vial is covered without an unmanageable number of duplicate/false vial proposals.

Blocked by:
- Waiting for Slurm job `290674` and whole-vial coverage results.

### 2026-10-28 — Whole-vial localization results

Completed:
- Inspected completed job `290674` and audited train/validation overlays.
- Measured one-to-one containment of reviewed cap/opening centers within whole-vial proposals.
- Evaluated full vial boxes and top-region fractions of 50, 35, and 25 percent.

Validated:
- Job `290674` completed with exit code `0:0` and correctly processed train plus validation only.
- LocateAnything produced 62 vial proposals for 75 reviewed vial objects.
- Best one-to-one top-region coverage was 62/75, or 82.7 percent.
- Missing-cap coverage was 4/10; six missing-cap vials remained unlocalized.
- Validation produced only 8 proposals for 15 reviewed vials.
- No image exceeded the 35-object anomaly limit.

Decisions:
- Reject zero-shot whole-vial localization as the production proposal policy; it does not meet the recall gate.
- Do not implement VLM crop classification on an incomplete proposal set.
- The next LocateAnything path should be use-case fine-tuning on reviewed vial/cap geometry rather than further zero-shot prompt iteration.

Next:
- Prepare LocateAnything LoRA training examples from the reviewed train split, including every missing-cap vial and loose table-cap example.
- Compare a fine-tuned vial localizer with a conventional high-recall detector/proposal baseline before committing the trainer to one proposal backend.

Blocked by:
- Neither zero-shot cap/opening prompts nor zero-shot vial prompts provide sufficient missing-cap recall.

### 2026-10-28 — Three-phase vial/cap association pilot

Completed:
- Added deterministic one-to-one vial/cap association with configurable upper-vial geometry.
- Added inferred top boxes for vials without matched caps and preserved unmatched caps for later loose-cap/Z handling.
- Added color-coded overlays and per-image association records.
- Ran the combiner over the train/validation whole-vial and visible-cap manifests.

Validated:
- The expanded focused suite passes 31 tests.
- Across four images: 62 vial proposals, 65 cap proposals, 58 matches, 4 inferred missing-cap tops, and 7 unmatched caps.
- All four inferred missing-cap boxes corresponded to true `missing_cap` annotations: 4/4 precision in this pilot.
- Missing-cap recall remained 4/10 because Phase 1 failed to localize six missing-cap vials.
- Visual inspection confirms the red inferred boxes are positioned over uncapped vial tops in the two frames where those vials were detected.

Decisions:
- Adopt vial-minus-cap association as the missing-cap inference design; it correctly inferred every unmatched detected vial in this pilot.
- Treat Phase 1 vial recall, not association logic, as the remaining blocker.
- Do not classify unmatched caps as table spares from 2D geometry alone; retain them as candidates for VLM labeling and runtime Z qualification.

Next:
- Fine-tune or replace the Phase 1 vial localizer to recover the six missed missing-cap vials.
- Re-run the unchanged association stage and require full missing-vial coverage before generating pseudo-label crops.

Blocked by:
- Phase 1 whole-vial localization covers only 4 of 10 missing-cap vials; association cannot infer a missing cap for a vial that was never proposed.

### 2026-10-28 — Dark uncapped-vial rim prompt experiment

Completed:
- Added an isolated localization experiment describing the visible dark circular rim/opening of an uncapped clear vial.
- Added two prompt variants that explicitly exclude empty metal-tray holes and yellow-capped vials.
- Submitted train/validation-only Slurm job `290675` to `dark_rim_proposals_train_val`.

Validated:
- The experiment configuration loads successfully with both prompts.
- The focused suite remains at 31 passing tests.
- `git diff --check` passes.
- Job `290675` entered the queue with a normal priority wait.

Decisions:
- Treat the dark rim as a visible localization feature, avoiding language centered on an absent cap.
- Keep this experiment independent of the validated vial-minus-cap association path until its missing-cap recall and tray-hole false-positive rate are measured.

Next:
- Measure each prompt variant separately against reviewed missing-cap boxes.
- Audit whether detections correspond to vial rims or empty tray apertures.
- Merge this stream with vial-minus-cap inference only if it adds true missing-cap recall without excessive false positives.

Blocked by:
- Waiting for Slurm job `290675` and dark-rim precision/recall results.

### 2026-10-28 — Dark-rim prompt results

Completed:
- Inspected completed Slurm job `290675` and separated results by prompt and image.

Validated:
- Job `290675` completed with exit code `0:0`.
- Both dark-rim prompt variants returned zero boxes on all four train/validation images.
- Missing-cap recall was 0/10, with no tray-hole false positives because no proposals were emitted.

Decisions:
- Reject both zero-shot dark-rim prompts.
- Stop further zero-shot wording iterations for missing-cap localization; three distinct formulations have failed to ground the feature.
- Retain vial-minus-cap association because its four inferred missing-cap boxes were all correct, and focus further work on supervised improvement of Phase 1 vial recall.

Next:
- Build LocateAnything LoRA examples for full-vial localization from the reviewed training split, or compare a conventional detector trained on those vial boxes.
- Reuse the existing association stage unchanged after improving vial proposal recall.

Blocked by:
- Zero-shot LocateAnything does not localize the dark uncapped-vial rim in these images.

### 2026-10-28 — Qwen3-VL-32B grounding comparison

Completed:
- Added a Qwen3-VL visual-grounding backend without changing the proposal manifest consumed by later crop and labeling stages.
- Added robust parsing for normalized `bbox_2d` JSON, including fenced output, clamping, invalid-entry rejection, and NMS.
- Added an isolated four-prompt train/validation experiment for full vials, visible caps, uncapped dark rims, and loose table caps.
- Submitted Slurm job `290677` using two H100 GPUs and BF16 model weights.

Validated:
- The shared inference environment has Transformers 4.57.1 and Accelerate 1.15.0.
- The expanded focused suite passes 34 tests.
- `git diff --check` passes.
- The cluster exposes both four- and eight-H100 nodes; job `290677` is queued normally for two H100s.

Decisions:
- Keep Qwen output in the same records, overlays, manifest, and report format as LocateAnything for a direct comparison.
- Limit this first run to the four train/validation images; preserve the test image for later unbiased evaluation.
- Use the same 35-object anomaly threshold and no automatic acceptance of Qwen labels.

Next:
- Inspect the per-prompt overlays and measure reviewed-object coverage after job `290677` completes.
- Run the existing vial-minus-cap association on Qwen proposals if full-vial recall improves.
- Decide whether Qwen should replace Phase 1, supplement LocateAnything, or be rejected before scaling to the remaining real images.

Blocked by:
- Waiting for Slurm job `290677` to receive two H100 GPUs and download/load the 32B checkpoint.
