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
- `crop_box_xyxy` is derived deterministically from the source box and crop configuration.
- An effective training label is accepted only when `status` is `accepted` or `human_corrected`.
- Raw model responses and human corrections remain available in provenance records.

## Implementation Checklist

### Milestone 0 — Establish a Reviewed Pilot Set

- [ ] Select a small but diverse set of real images covering capture sessions, lighting, seated caps, crooked caps, missing caps, and loose table caps.
- [ ] Define the grouping key for the available real images.
- [ ] Assign pilot train, validation, and test groups before labeling.
- [ ] Manually label all objects in the pilot test set.
- [ ] Manually label enough training examples to provide several references per class.
- [ ] Confirm loose intact table caps use `good_cap`.
- [ ] Record the pilot source and grouping policy in a source manifest.

Acceptance criteria:

- Every pilot image belongs to exactly one split group.
- No split group crosses train, validation, and test.
- The pilot test set is fully reviewed and contains representative examples of every class being scored.

### Milestone 1 — Configuration, Schemas, and Box Utilities

- [ ] Add a versioned YAML project configuration schema.
- [ ] Validate class names, unique IDs, crop scale, thresholds, and split fractions.
- [ ] Add typed representations for images, proposals, classifications, and corrections.
- [ ] Implement YOLO-to-pixel and pixel-to-YOLO conversions.
- [ ] Implement centered 1.25-times box expansion with image-boundary clipping.
- [ ] Implement stable object IDs independent of processing order.
- [ ] Add JSONL readers and atomic record writers with schema-version checks.
- [ ] Add tests for box expansion, clipping, degenerate boxes, and coordinate round trips.

Acceptance criteria:

- Unit tests prove exact crop behavior for centered, edge, corner, one-pixel, and invalid boxes.
- Source geometry is immutable after record creation.
- Configuration errors fail before any dataset artifacts are written.

### Milestone 2 — Seed Annotation Import

- [ ] Accept a standard YOLO seed dataset and project configuration.
- [ ] Verify that images, labels, and class IDs agree with the configuration.
- [ ] Convert seed labels into canonical object records.
- [ ] Generate 1.25-times reference crops.
- [ ] Generate per-class contact sheets and full-image overlays.
- [ ] Reject seeds outside the training split from the VLM reference pool.
- [ ] Write a seed manifest containing provenance and class counts.

Acceptance criteria:

- Every imported box appears in both a canonical record and an audit overlay.
- Reference crops are sourced only from training groups.
- Importing the same source twice produces identical object IDs and crop geometry.

### Milestone 3 — Generic LocateAnything Proposals

- [ ] Extract reusable worker and response-parsing logic from the existing LocateAnything evaluation scripts without breaking them.
- [ ] Accept an unlabeled image directory, configuration, split manifest, and output directory.
- [ ] Use one configurable object-of-interest prompt by default.
- [ ] Preserve raw responses, model identity, adapter identity, seed, and inference settings.
- [ ] Clamp coordinates and reject degenerate predictions.
- [ ] Apply configurable NMS or proposal clustering.
- [ ] Resume at image granularity.
- [ ] Record images with zero proposals.
- [ ] Flag unusual proposal counts without assuming a fixed aperture grid.
- [ ] Generate full-image proposal overlays.
- [ ] Add a Slurm launcher for full proposal generation.

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
| 0. Reviewed pilot set | Not started | — |
| 1. Configuration and schemas | Not started | — |
| 2. Seed import | Not started | — |
| 3. LocateAnything proposals | Not started | Existing specialized scripts available for refactoring |
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
