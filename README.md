# Geocoding Box

Runbook for day-to-day commands:

```text
HOW_TO_RUN_GEOCODING.md
```

This folder now has one stable no-AI workflow and one no-API address-point
workflow.  The address-point workflow treats the `1_address_to_point_gemini`
GPKG as the primary case layer, and can optionally reuse the JSON/JSONL output
from `spatial_capture_production/1_address_to_point_gemini.py` as supplemental
evidence:

```text
case address/OCR/local-OS evidence -> ranked no-AI candidate boxes
ranked boxes + local base polygons -> ranked no-AI polygon candidates
```

It does not call Gemini, DeepSeek, OpenAI, or any external model API.  The
pipeline uses council-profile paths, OCR JSONL, OS OpenNames, OS OpenRoads,
and local base polygon layers.  Truth/evaluation layers are only used when you
explicitly pass them.

The upstream script is treated as a black box and is not invoked or modified by
geocoding-box.  Case identity comes from the GPKG `unique_key`; JSON fields such
as `_source_unique_key` are never used to create the case set.

## Directory Layout

```text
geocoding-box/
  run_no_ai_candidate_boxes.py              # preferred entrypoint
  run_address_point_council_pipeline.py     # 1_address GPKG + optional JSONL -> boxes -> council parcels
  17_run_no_gemini_full_a_pipeline.py       # compatibility wrapper
  no_ai_candidate_boxes/
    pipeline.py                             # orchestrates all active stages
    polygon_ranking.py                      # ranks in-box polygon candidates
    stages/
      build_input_candidates.py
      ocr_candidate_rerank.py
      openname_anchor.py
      openroads_relation.py
      plan_ocr_anchor_audit.py
      evidence_roi.py
      address_point_evidence_adapter.py
      multi_roi_candidates.py
      corridor_augmented_roi.py
  tools/
    build_mansfield_merged_basemap.py
  legacy_experiments/
    README_legacy_notes.md
    *.py                                    # older experiments/diagnostics
  tmp_results/                              # local experiment outputs
```

## Run Polygon Ranking

The polygon ranking stage starts from a polygon-candidate CSV emitted by the
hybrid WFS/council candidate generator.  It does not call any AI/API service.
Truth columns are used only when present for offline evaluation.

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/no_ai_candidate_boxes/polygon_ranking.py \
  --input-csv /path/to/polygon_candidates_top100_or_top200.csv \
  --output-prefix /path/to/polygon_ranking_output \
  --strategy relation_evidence_mix \
  --candidate-source-filter council_cadastral \
  --top-k 5 \
  --output-top-n 100
```

Current deterministic strategies:

```text
rule                 keep the existing rule_rank order
conservative_road    preserve rule top4, then low-confidence-only road/ROI alternate
portfolio            preserve N rule candidates, then add non-cover/road alternates
bounded_noncover     preserve rule top4, then add one bounded non-cover alternate for low-confidence relation cases
diverse_footprint    preserve rule top1, then avoid near-duplicate footprint candidates in top5
relation_evidence_mix preserve rule top3 for relation/road-only cases, then add evidence/corridor alternates
centroid_cluster     group nearby candidate polygons into one union-style candidate; diagnostic only, not atomic
```

Recent Mansfield checks:

```text
seed48 top5: rule 41/50 -> conservative_road 43/50
random1282 top5: rule 1098/1282 -> conservative_road 1099/1282
random1282 top10: rule 1127/1282 -> portfolio top10 1138/1282
random1282 top5: rule 1098/1282 -> bounded_noncover 1098/1282
heldout seed51-53 top5: rule 106/150 -> bounded_noncover 109/150
seed49 top5: rule 78/99 -> bounded_noncover 78/99
random1282 top5: rule 1098/1282 -> diverse_footprint 1106/1282
heldout seed51-53 top5: rule 106/150 -> diverse_footprint 113/150
seed49 top5: rule 78/99 -> diverse_footprint 81/99
random1282 atomic top5: rule 1080/1282 -> relation_evidence_mix 1105/1282
GPKG-primary seed6201 boxes top5: 46/50 -> fuzzy-road midpoint 48/50
GPKG-primary seed6201 council atomic top5: 38/50 -> relation_evidence_mix 41/50
random1282 top5: centroid_cluster80 1201/1282
heldout seed51-53 top5: centroid_cluster80 137/150
seed49 top5: centroid_cluster80 92/99
blind seed54 top5: centroid_cluster80 85/100
blind seed55 top5: centroid_cluster80 91/100
```

## Run The Current No-AI Box Pipeline

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_no_ai_candidate_boxes.py \
  --truth-gpkg /path/to/input_or_truth_sample.gpkg \
  --truth-layer layer_name \
  --output-prefix /path/to/output_prefix
```

The default council profile is:

```text
configs/mansfield.json
```

The explicit Mansfield form is still useful when you want the command to be
self-documenting:

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_no_ai_candidate_boxes.py \
  --config /env/code/spatial-capture-monmouthshire/geocoding-box/configs/mansfield.json \
  --truth-gpkg /path/to/input_or_truth_sample.gpkg \
  --truth-layer layer_name \
  --output-prefix /path/to/output_prefix
```

To add another council, copy:

```text
configs/template_council.json
```

and edit these fields first:

```text
council_root
local_bbox
locality_names
allowed_locality_names
generic/openname/plan generic place names
paths.*
layers.*
parameters.roi_side / evidence_roi_count if needed
```

You can also override the local directory without editing the JSON:

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_no_ai_candidate_boxes.py \
  --config /path/to/my_council.json \
  --council-root /data/my-council \
  --truth-gpkg /path/to/input.gpkg \
  --truth-layer layer_name \
  --output-prefix /path/to/output_prefix
```

Useful current defaults:

```text
--roi-side 180
--evidence-roi-count 5
--corridor-step-m 50
--corridor-pad-m 180
```

Clean outputs:

```text
<output-prefix>_no_ai_input_candidates.json
<output-prefix>_no_ai_case_summary.csv
<output-prefix>_no_ai_candidate_boxes.csv
<output-prefix>_no_ai_manifest.json
```

The intermediate legacy-compatible files are still written because some stages
share an older schema.  In particular, a file with `gemini_like` in its name is
only a schema-compatible local JSON artifact; it is not a Gemini/API call.

## Run 1_Address-To-Council-Parcel Pipeline

Use this when `1_address_to_point_gemini.py` should replace the old
geocoder-like front stage.  The upstream script itself is not edited; the new
adapter uses the 1_address/input GPKG as the case base and consumes JSON/JSONL
only as optional evidence.

Reusing an existing address-point GPKG plus optional JSON/JSONL evidence:

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_address_point_council_pipeline.py \
  --config /env/code/spatial-capture-monmouthshire/geocoding-box/configs/mansfield.json \
  --address-point-gpkg /path/to/1_address_to_point_gemini.gpkg \
  --address-point-layer layer_name \
  --address-point-json /path/to/1_address_output.jsonl \
  --output-prefix /path/to/output_prefix
```

Main outputs:

```text
<output-prefix>_input_candidates.json
<output-prefix>_candidate_boxes.csv
<output-prefix>_candidate_boxes.gpkg
<output-prefix>_polygon_candidates_top200.csv
<output-prefix>_council_atomic_top5_selected_top5.csv
<output-prefix>_pipeline_manifest.json
```

`--address-point-gpkg` / `--address-point-layer` are the local case metadata
layer used by the OCR/OpenNames/OpenRoads/plan-audit stages.  They provide the
authoritative `unique_key` and geometry.  For offline evaluation only, also add
`--truth-gpkg` and `--truth-layer`; accuracy fields are meaningful only when
those truth geometries are real target polygons.  The old `--case-gpkg` and
`--truth-gpkg` pairs are still accepted as legacy aliases, but the current
workflow should start from `--address-point-gpkg`.

Box generation mode defaults to `--box-mode auto`:

```text
full-a  when --address-point-gpkg/--address-point-layer are supplied; keeps OCR/OpenNames/OpenRoads/plan-audit stages
direct  when no case geometry is supplied; uses only 1_address evidence + OpenRoads
```

## Run Direct Address-To-Point Geocoding

For a single address, or address plus OCR text, use:

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/geocode_address_point.py \
  --address "4-6 Leeming Street, Mansfield" \
  --pretty
```

With OCR road/geocode evidence:

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/geocode_address_point.py \
  --address "2911 Field Mill" \
  --ocr-text "NOTTINGHAM ROAD | QUARRY LANE" \
  --pretty
```

This returns one EPSG:27700 point:

```text
easting_27700
northing_27700
source
matched_address
confidence
candidates
```

The implementation is:

```text
geocode_address_point.py
no_ai_candidate_boxes/address_point_geocoder.py
```

## Active Stage Map

The main orchestrator is:

```text
no_ai_candidate_boxes/pipeline.py
```

It runs these stages in order:

```text
build_input_candidates.py       local text/OCR/OS evidence -> geocoder-like JSON
ocr_candidate_rerank.py         OCR grid/geocode correction evidence
openname_anchor.py              OS OpenNames anchors from OCR/address text
openroads_relation.py           OpenRoads relation/line evidence
plan_ocr_anchor_audit.py        selected plan-page OCR anchor audit
multi_roi_candidates.py         ranked evidence boxes
corridor_augmented_roi.py       road-corridor boxes + final box set
```

## Current Accuracy Notes

On held-out Mansfield random100 checks after removing the API stage:

```text
seed45 top5: 96/100
seed46 top5: 93/100
seed47 top5: 96/100
```

So the honest current expectation for fully automatic no-AI candidate-box
intersection recall is about 93-96% on Mansfield-style samples, with top5 being
the main target metric.

## Legacy Material

Older Monmouthshire box experiments, Gemini/OS/GOG union runs, WFS polygon
ranking experiments, selector training scripts, and visualization helpers are
kept under:

```text
legacy_experiments/
```

The original long notes were preserved at:

```text
legacy_experiments/README_legacy_notes.md
```

Use those files for archaeology or comparison only.  New changes to the current
box-generation flow should normally go under `no_ai_candidate_boxes/`.
