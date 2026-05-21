# Mansfield ROI Union Pipeline

## Monmouthshire Evidence-to-Box MVP

`evidence_to_box_pipeline.py` is the first Monmouthshire target-oriented box
generator.  It avoids LLM/Gemini/DeepSeek best-address selection and generates
ranked candidate boxes from production-visible evidence:

- current input geometry as optional context/protected prior;
- address road/name/number parsing;
- OpenRoads named road geometry;
- OS OpenNames local anchors;
- Monmouthshire WFS Land/Building polygons near evidence anchors.

Run the 200-case experiment without protected current boxes:

```bash
python /env/code/spatial-capture-monmouthshire/geocoding-box/evidence_to_box_pipeline.py \
  --input-gpkg /data/monmouthshire/spatial/base-map/tmp_output/monmouthshire_input_layer_random200_seed42_20260505.gpkg \
  --input-layer features \
  --truth-gpkg /data/monmouthshire/spatial/base-map/tmp_output/monmouthshire_capture_random200_seed42_merge_direct_20260505.gpkg \
  --truth-layer capture_result \
  --no-current-boxes \
  --output-prefix /data/monmouthshire/spatial/base-map/tmp_output/monmouthshire_evidence_boxes_random200_seed42_v1_no_current_boxes
```

Current proxy evaluation against the random200 WFS capture result:

- top1 box intersects truth proxy: 181/200
- top3: 195/200
- top5: 197/200
- top10: 198/200
- top20: 200/200
- union: 200/200
- median boxes per case: 37.5

Primary outputs:

```text
/data/monmouthshire/spatial/base-map/tmp_output/
  monmouthshire_evidence_boxes_random200_seed42_v1_no_current_boxes_boxes.gpkg
  monmouthshire_evidence_boxes_random200_seed42_v1_no_current_boxes_boxes.csv
  monmouthshire_evidence_boxes_random200_seed42_v1_no_current_boxes_case_summary.csv
```

This folder contains the Mansfield parcel candidate ROI workflow that produced
the current protected-current + evidence + road-corridor union baseline.

Default one-command run:

```bash
python /env/code/spatial-capture-monmouthshire/geocoding-box/run_roi_union_pipeline.py
```

Default behavior:

- sample 1000 cases from the Mansfield full GPKG, excluding the existing
  random200 sample;
- run OS/Gemini/GOG address candidates;
- run OCR grid correction;
- run OpenNames/OpenRoads conservative refinements;
- audit plan OCR road and feature anchors;
- generate protected-current + evidence ROIs;
- add road-corridor ROIs;
- combine with the existing random200 baseline.

Current baseline output:

```text
/data/mansfield/spatial/polygon-layer/tmp_output/
  mansfield-manual-polygon-link_random1200_seed42_43_combined_corridor_augmented_roi_step50_v1.csv
```

Current baseline metrics on evaluable expanded rows:

- polygon intersects ROI union: 1216/1282
- hit rate: 94.85%
- median candidate polygon count: 64

## No-AI Mansfield Candidate Boxes

Use `run_no_ai_candidate_boxes.py` as the clean production-facing entrypoint.
It wraps the legacy runner `17_run_no_gemini_full_a_pipeline.py`, which is kept
for compatibility with older experiment files.  The workflow uses Mansfield
textual OS results, OCR grid candidates, OpenNames/OpenRoads anchors, plan OCR
road audit, multi-ROI selection, and corridor boxes.  It does not call Gemini,
DeepSeek, OpenAI, or any external model API.

The current default setting is `--roi-side 180` and
`--evidence-roi-count 5`.  The seed47 held-out random100 run reached top5
ROI-box intersection `96/100`; seed46 reached `93/100`, so the honest expected
range is still about 93-96%.

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_no_ai_candidate_boxes.py \
  --truth-gpkg /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_random100_excl1200_500_seed45_truth.gpkg \
  --truth-layer random100_excl1200_500_seed45 \
  --output-prefix /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_random100_no_ai_boxes
```

Clean outputs from the wrapper:

```text
<output-prefix>_no_ai_input_candidates.json
<output-prefix>_no_ai_case_summary.csv
<output-prefix>_no_ai_candidate_boxes.csv
<output-prefix>_no_ai_manifest.json
```

Useful commands:

```bash
# Reuse existing artifacts and rebuild combined outputs.
python geocoding-box/run_roi_union_pipeline.py

# Force a full rerun.
python geocoding-box/run_roi_union_pipeline.py --no-resume

# Run only the new sampled set and skip combining with random200.
python geocoding-box/run_roi_union_pipeline.py --skip-combine

# Change sample size and seed.
python geocoding-box/run_roi_union_pipeline.py --sample-size 2000 --seed 44
```

Visualization helpers:

```bash
python geocoding-box/7_visualize_rescued_roi_context.py
python geocoding-box/7_visualize_rescued_roi_cases.py
```

Production-visible base-layer reranking:

```bash
python geocoding-box/8_candidate_base_layer_rerank_experiment.py
```

This expands every ROI union into WFS topographic polygon candidates from:

```text
/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg
```

and uses OS Open UPRN points from:

```text
/data/base-data/osopenuprn_202602.gpkg
```

as supporting point evidence.  The manual polygon-link layer is used only as
offline truth for evaluation; it is not a production candidate source.

Outputs:

```text
/data/mansfield/spatial/polygon-layer/tmp_output/
  mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v1_cases.csv
  mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v1_top20.csv
  mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v1_candidates.csv
```

Current base-layer v2 metrics use the production-visible filter
`Theme contains Land|Building`.  Offline evaluation hit means "candidate
intersects manual truth polygon":

- WFS candidate set contains a truth-intersecting polygon: 1217/1280
- top1: 905/1280
- top3: 983/1280
- top5: 1013/1280
- top10: 1057/1280
- median WFS candidate count per case: 428

Build Mansfield WFS land/building merged basemap:

```bash
python geocoding-box/build_mansfield_merged_basemap.py
```

This uses the current `capture_wfs_merge.py` shared-edge logic to merge small
WFS `Land` polygons into adjacent `Buildings`, and writes:

```text
/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg
  layer: mansfield_polygons_in_buffers_merged
```

Validation against the random1200 manual polygon sample also compares
Mansfield cadastral parcels from:

```text
/data/mansfield/spatial/base-map/mansfield_councils_land.gpkg
```

The cadastral layer is read dynamically by bbox during validation/ranking,
not copied into the merged GPKG, because `/data` is nearly full.

Validation result on 1200 manual polygons, using best intersecting visible
candidate per target:

- raw WFS Land/Building median best IoU: 0.647
- WFS land/building merged median best IoU: 0.750
- council cadastral parcels median best IoU: 0.809
- best of raw WFS + WFS merged + cadastral median best IoU: 0.9996

Conclusion: polygon ranking should use a hybrid production-visible candidate
pool: raw WFS Land/Building + WFS merged + bbox-filtered council cadastral
parcels.  The manual polygon-link layer remains evaluation-only.

Range-anchor reranking:

```bash
python geocoding-box/8_candidate_base_layer_rerank_experiment.py \
  --output-prefix /data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v5_range_anchor_strong_theme_land_building
```

This version adds a production-visible range anchor extracted from existing
OS/GOG candidates for addresses such as `4-6 Leeming Street`,
`56-58 Westfield Lane`, and `Land Between 109-115 Westfield Lane`.  Complete
range anchors can override a conflicting old best/v7 point; partial anchors
remain weak evidence.

Current v5 metrics:

- WFS candidate set contains a truth-intersecting polygon: 1217/1280
- top1: 928/1280
- top3: 997/1280
- top5: 1023/1280
- top10: 1058/1280
- top20 after compression: 1093/1280
- top50 after compression: 1135/1280

Compressed production review pool:

```bash
python geocoding-box/8_compress_base_layer_candidates.py
```

This keeps the top 50 WFS candidates per case from the v2 score ranking and
also writes a top20 review file:

```text
/data/mansfield/spatial/polygon-layer/tmp_output/
  mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v3_compressed_top50_cases.csv
  mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v3_compressed_top50_top20.csv
  mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v3_compressed_top50_top50.csv
```

Compression keeps top1/top10 unchanged while reducing the median candidate
count from 428 to 50:

- top1: 905/1280
- top10: 1057/1280
- top20: 1088/1280
- top50: 1131/1280

Compress the range-anchor v5 output:

```bash
python geocoding-box/8_compress_base_layer_candidates.py \
  --input-candidates /data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v5_range_anchor_strong_theme_land_building_candidates.csv \
  --output-prefix /data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v6_range_anchor_compressed_top50 \
  --max-rank 50
```

Visualize selected or random top20 cases:

```bash
python geocoding-box/8_visualize_base_layer_top20.py \
  --top20-csv /data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v6_range_anchor_compressed_top50_top20.csv \
  --cases-csv /data/mansfield/spatial/polygon-layer/tmp_output/mansfield-manual-polygon-link_random1200_seed42_43_combined_base_layer_rerank_v6_range_anchor_compressed_top50_cases.csv \
  --out-dir /data/mansfield/spatial/polygon-layer/tmp_output/base_layer_top20_v6_range_anchor_examples \
  --case-keys 1569,1878,2885,6977,3446
```

There is also a diagnostic-only script,
`8_candidate_polygon_rerank_experiment.py`, that ranks against the manual
polygon-link layer itself.  Do not use that script as the production candidate
source; it leaks the hidden evaluation layer by design.

Hybrid production candidate pool and cascade selector:

```bash
python geocoding-box/9_hybrid_polygon_rerank.py
python geocoding-box/10_cascade_polygon_selector.py
```

`9_hybrid_polygon_rerank.py` builds the production-visible candidate pool from
raw WFS Land/Building, WFS merged Land/Building, and bbox-filtered council
cadastral parcels.  `10_cascade_polygon_selector.py` is a post-processor for
that top50 pool.  It keeps the current rule top1 for relation/weak cases, but
for strong exact OS address cases it uses saved OS/GOG candidate address points
to prefer polygons that contain the requested address points and fewer unrelated
address points.

Latest selector outputs:

```text
/data/mansfield/spatial/polygon-layer/tmp_output/
  mansfield-manual-polygon-link_random1200_seed42_43_combined_cascade_polygon_selector_v1_selected.csv
  mansfield-manual-polygon-link_random1200_seed42_43_combined_cascade_polygon_selector_v1.xlsx
  mansfield-manual-polygon-link_random1200_seed42_43_combined_cascade_polygon_selector_v1.summary.json
```

Offline diagnostic metrics on the 1282 evaluable random1200/seed42+43 rows:

- previous hybrid top1 IoU>=0.5: 611/1282
- cascade selector IoU>=0.5: 643/1282
- previous hybrid top1 IoU>=0.8: 498/1282
- cascade selector IoU>=0.8: 532/1282
- exact address selector subset IoU>=0.5: 451/593
- current-rule fallback subset IoU>=0.5: 192/687

This is a modest exact-address improvement, not a full polygon solution.  The
remaining hard cases are mostly relation/parcel semantics (`land`, `rear`,
`adjacent`, `plot`, `between`) where the address point is an anchor rather than
the target polygon.

Composite/relation candidate experiment:

```bash
python geocoding-box/11_composite_relation_selector.py
```

The script generates case-local composite candidates from the hybrid top-N pool:

- exact-address unions from polygons carrying requested OS/GOG address points
- touching same-source unions
- relation-near-anchor unions for `land/rear/adjacent/plot/between` cases

By default it does **not** allow composites to replace the cascade-selected top1,
because the first promotion heuristic caused many false promotions.  Use
`--promote-composites` only for diagnostic experiments.

Top50 composite/relation v1:

- composite rows generated: 5279
- oracle IoU>=0.5: 953/1282 = 74.34%
- selected IoU>=0.5 with promotion enabled: 562/1282, worse than cascade

Top200 hybrid pool, written under `geocoding-box/tmp_results/` to avoid filling
`/data`:

```bash
python geocoding-box/9_hybrid_polygon_rerank.py \
  --max-output-rank 200 \
  --output-prefix /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_random1200_hybrid_top200_v1

python geocoding-box/11_composite_relation_selector.py \
  --top-csv /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_random1200_hybrid_top200_v1_top200.csv \
  --output-prefix /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_random1200_composite_relation_top200_v2_no_promotion
```

Top200 diagnostics:

- single-candidate top200 oracle IoU>=0.5: 1003/1282 = 78.24%
- top200 + composite oracle IoU>=0.5: 1015/1282 = 79.17%
- full uncapped hybrid candidate oracle IoU>=0.5: 1025/1282 = 79.95%
- no-promotion selected IoU>=0.5: 642/1282 = 50.08%

Conclusion: composites help candidate recall only modestly in the current form.
The hard blocker for an 80% automatic top1 result is still selection logic for
the 300+ cases where a good candidate exists but is not selected, especially
relation/parcel semantics where the address point is only an anchor.

## Mansfield In-Box Polygon Selector Experiments

These are offline experiments on the 1282 Mansfield rows with manual polygon
truth.  They do not call AI/API services during scoring; truth columns are used
only for labels/evaluation.

```bash
python geocoding-box/13_train_polygon_selector.py
python geocoding-box/14_spatial_diverse_polygon_selector.py --min-centroid-distance-m 2
python geocoding-box/15_hgb_polygon_selector_experiment.py
```

Current results:

- baseline hybrid polygon Top1 intersects: 988/1282 = 77.07%
- baseline hybrid polygon Top5 intersects: 1098/1282 = 85.65%
- pairwise linear ranker: did not beat the baseline; keep as diagnostic only
- spatial-diverse Top5 selector: 1110/1282 = 86.58%
- 5-fold OOF HGB Top1 selector: 999/1282 = 77.93%
- HGB Top1 + spatial-diverse Top5 combo: 1115/1282 = 86.97%

Interpretation: the strongest immediate production change is to keep the current
rule score as the main candidate generator, add a small spatial-diversity pass
for the review Top5, and treat HGB as a possible top1 suggestion rather than a
replacement for the high-recall Top5 set.  The remaining gains need stronger
parcel-relation evidence from the building/parcel plan images (`rear`,
`adjacent`, `junction`, `between`, `frontage`) rather than more generic
candidate-level reweighting.
