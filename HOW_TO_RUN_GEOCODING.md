# Geocoding Runbook

这份文档只记录当前还在用的流程，避免后面被旧实验脚本和旧文件名绕晕。

当前默认 council 是 **Mansfield**，默认配置文件是：

```text
configs/mansfield.json
```

当前候选框主流程是 **no-AI / no-API**：

- 不调用 Gemini
- 不调用 DeepSeek
- 不调用 OpenAI
- `gemini_like` 只是旧 schema 兼容文件名，不代表 API 调用

工作目录建议固定为：

```bash
cd /env/code/spatial-capture-monmouthshire/geocoding-box
```

Python 解释器固定用：

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python
```

## 1. 地址到坐标

目标：

```text
输入：地址，或者 地址 + OCR 文本
输出：一个 EPSG:27700 坐标点
```

入口脚本：

```text
geocode_address_point.py
```

实现主体：

```text
no_ai_candidate_boxes/address_point_geocoder.py
```

### 只用地址

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/geocode_address_point.py \
  --address "4-6 Leeming Street, Mansfield" \
  --pretty
```

输出里最重要的字段：

```text
easting_27700
northing_27700
source
matched_address
confidence
candidates
```

例子：

```text
address = 4-6 Leeming Street, Mansfield
source = global_text_os
matched_address = 4-6, LEEMING STREET, MANSFIELD, NG18 1NE
confidence = 98.0
```

### 地址 + OCR 文本

OCR 文本可以放路名、地物名、表格里的 Geo Code、location line 等。

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/geocode_address_point.py \
  --address "2911 Field Mill" \
  --ocr-text "NOTTINGHAM ROAD | QUARRY LANE" \
  --pretty
```

这个例子会把 OCR 里的两条路作为锚点，选：

```text
NOTTINGHAM ROAD & QUARRY LANE
```

附近的交叉重心作为坐标点。

如果 OCR 文本很多，写入一个 `.txt` 后这样跑：

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/geocode_address_point.py \
  --address "2911 Field Mill" \
  --ocr-text-file /path/to/ocr_text.txt \
  --pretty
```

### 单点准确度口径

最近一次 seed49 随机 100 测试结果保存在：

```text
tmp_results/mansfield_seed49_address_point_geocoder_eval_v1.csv
tmp_results/mansfield_seed49_address_point_geocoder_eval_v1.summary.json
```

结果大概是：

```text
address only:
直接落入真实 polygon: 55/99 = 55.56%
150m 内: 82/99 = 82.83%

address + OCR:
直接落入真实 polygon: 58/99 = 58.59%
150m 内: 83/99 = 83.84%
```

注意：这个是“单点”准确度，不是候选框召回率。单点只能作为定位锚点，不能稳定直接替代最终地块 polygon。

### Gemini/OS 地址点批处理

这是会产生 API 成本的上游入口。Mansfield 示例：

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/spatial_capture_production/1_address_to_point_gemini.py \
  --council-profile mansfield \
  --input-gpkg /data/mansfield/spatial/polygon-layer/mansfield-manual-polygon-link.gpkg \
  --input-layer mansfield-manual-polygon-link \
  --output-jsonl /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_address_to_point_gemini.jsonl \
  --output-xlsx /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/mansfield_address_to_point_gemini.xlsx
```

Mansfield 默认字段：

```text
--county Nottinghamshire
--council Mansfield
--address-column chargegeog
--input-layer mansfield-manual-polygon-link
--gemini-model gemini-3-flash-preview
--llm-execution-mode batch
--gemini-batch-size 1000
```

Batch/resume 规则：

```text
默认开启断点续传。
如果 output JSONL 已存在，里面已有 best_source_final 的 row 会跳过。
Batch prompts/job/raw results 会写到同目录 sidecar：
  *_batch_prompts.json
  *_batch_job.json
  *_batch_raw_results.json
超过 --gemini-batch-size 的大任务会拆成多个 *_batchpart0001_* sidecar。
重跑时只要 prompt_signature 一致，就会复用已有 Batch job 或 raw results，不会重复提交同一批 Gemini 请求。
```

如果要临时改回实时 API：

```text
--llm-execution-mode realtime
```

如果以后要跑 Monmouthshire，再显式切 profile 和对应输入：

```text
--council-profile monmouthshire
```

### 把 1_address 接入 geocoding-box

现在不用改 `spatial_capture_production/1_address_to_point_gemini.py` 本体。
geocoding-box 以 `1_address_to_point_gemini.gpkg` 作为 case 主体，`unique_key`
和几何都从这个 GPKG 来；它已经生成好的 JSON/JSONL 只作为可选补充证据。
geocoding-box 自己不会再调用 Gemini / DeepSeek / Google / OS API。

如果已经有 `1_address` 的 GPKG 和 JSON/JSONL 输出：

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_address_point_council_pipeline.py \
  --config /env/code/spatial-capture-monmouthshire/geocoding-box/configs/mansfield.json \
  --address-point-gpkg /path/to/1_address_to_point_gemini.gpkg \
  --address-point-layer layer_name \
  --address-point-json /path/to/1_address_output.jsonl \
  --output-prefix /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_address_point_pipeline
```

`--address-point-gpkg` / `--address-point-layer` 是当前主流程的必填主体输入。
`--address-point-json` 可以传，也可以不传；传了以后只是补充 geocoder / OS /
Google 等上游证据，不负责决定 case 集合，也不负责修补 `unique_key`。

主要输出：

```text
<output-prefix>_input_candidates.json
<output-prefix>_candidate_boxes.csv
<output-prefix>_candidate_boxes.gpkg
<output-prefix>_polygon_candidates_top200.csv
<output-prefix>_council_atomic_top5_selected_top5.csv
<output-prefix>_pipeline_manifest.json
```

`--address-point-gpkg` / `--address-point-layer` 是本地证据链需要的 case 图层，
用来接 OCR、OpenNames、OpenRoads、plan-audit 等阶段。只有当你要做 Mansfield
离线评估时，才额外传 `--truth-gpkg` / `--truth-layer`，命中率字段才有意义。
旧命令里直接传 `--case-gpkg` 或 `--truth-gpkg` 也还兼容，但当前复现和生产
都应该从 `--address-point-gpkg` 开始。

`run_address_point_council_pipeline.py` 默认 `--box-mode auto`：

```text
有 --address-point-gpkg/--address-point-layer: 走 full-A，1_address 只替换原 build_input_candidates 前置段，OCR/OpenNames/OpenRoads/plan-audit 继续参与。
没有 case geometry: 走 direct，只用 1_address 输出 + OpenRoads 直接产候选框。
```

## 2. 地址到候选框

目标：

```text
输入：case GPKG
输出：每个 case 的多个候选框 ROI
```

入口脚本：

```text
run_no_ai_candidate_boxes.py
```

主调度：

```text
no_ai_candidate_boxes/pipeline.py
```

当前默认配置：

```text
configs/mansfield.json
```

### 运行候选框全流程

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_no_ai_candidate_boxes.py \
  --config /env/code/spatial-capture-monmouthshire/geocoding-box/configs/mansfield.json \
  --truth-gpkg /path/to/input_or_truth_sample.gpkg \
  --truth-layer layer_name \
  --output-prefix /env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_no_ai_boxes
```

参数说明：

```text
--truth-gpkg
  旧参数名。实验时传真实 polygon sample；生产时可以理解为 case input GPKG。

--truth-layer
  GPKG layer 名。

--output-prefix
  输出文件前缀。
```

输入 GPKG 至少应有：

```text
unique_key
chargegeog
FilePath
geometry
```

`geometry` 在离线实验里是真实 polygon，用于计算命中率；生产时如果没有真实 polygon，命中率字段没有意义，但候选生成逻辑本身不应依赖真实 polygon。

### 候选框主要输出

干净输出：

```text
<output-prefix>_no_ai_input_candidates.json
<output-prefix>_no_ai_case_summary.csv
<output-prefix>_no_ai_candidate_boxes.csv
<output-prefix>_no_ai_manifest.json
```

其中最常看的：

```text
<output-prefix>_no_ai_candidate_boxes.csv
```

一行是一个候选框，主要字段：

```text
case_key
base_key
roi_rank
roi_reason
roi_sources
roi_score
roi_minx
roi_miny
roi_maxx
roi_maxy
roi_center_easting
roi_center_northing
```

### 中间文件

后续候选 polygon 生成会用到这些中间文件：

```text
<output-prefix>_geocoder_gemini_like.json
<output-prefix>_openroads.csv
<output-prefix>_corridor_step50.csv
<output-prefix>_corridor_step50_rois.csv
```

再次提醒：

```text
*_gemini_like.json 不是 Gemini API 输出，只是兼容旧 schema 的本地 no-AI JSON。
```

### 候选框准确度口径

Mansfield held-out random100 的大致结果：

```text
top5 candidate boxes: 约 93%-96%
top8 candidate boxes: 约 95% 左右
```

这是“多个候选框里有没有一个和真实 polygon 相交”，不是 top1，也不是单点。

## 3. 地址到候选多边形

候选多边形流程分两步：

```text
候选框 ROI -> 从 WFS/council/base polygon layers 取候选 polygon -> polygon ranking
```

候选 polygon 的生产可见底图来源：

```text
/data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg
/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg
/data/mansfield/spatial/base-map/mansfield_councils_land.gpkg
/data/base-data/osopenuprn_202602.gpkg
```

真实 manual polygon 只用于离线评估，不能作为候选来源。

### 3.1 准备 merged WFS

如果这个文件已经存在，可以跳过：

```text
/data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg
```

如果缺失，运行：

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/tools/build_mansfield_merged_basemap.py \
  --wfs-gpkg /data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg \
  --wfs-layer mansfield_polygons_in_buffers \
  --council-gpkg /data/mansfield/spatial/base-map/mansfield_councils_land.gpkg \
  --council-layer cadastral_parcels \
  --output-gpkg /data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg \
  --output-layer mansfield_polygons_in_buffers_merged \
  --skip-validation
```

### 3.2 从候选框生成候选 polygon CSV

目前候选 polygon 生成脚本还在 legacy 目录里，但仍然是现在可用的生产可见候选生成器：

```text
legacy_experiments/9_hybrid_polygon_rerank.py
```

示例命令，假设前一步候选框输出前缀是：

```text
BOX_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_no_ai_boxes
```

运行：

```bash
BOX_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_no_ai_boxes
POLY_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_polygon_candidates

/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/legacy_experiments/9_hybrid_polygon_rerank.py \
  --input-json "${BOX_PREFIX}_geocoder_gemini_like.json" \
  --v10-csv "${BOX_PREFIX}_openroads.csv" \
  --case-summary-csv "${BOX_PREFIX}_corridor_step50.csv" \
  --rois-csv "${BOX_PREFIX}_corridor_step50_rois.csv" \
  --truth-gpkg /path/to/input_or_truth_sample.gpkg \
  --truth-layer layer_name \
  --raw-wfs-gpkg /data/mansfield/spatial/base-map/mansfield_wfs_polygon.gpkg \
  --raw-wfs-layer mansfield_polygons_in_buffers \
  --merged-wfs-gpkg /data/mansfield/spatial/base-map/mansfield_wfs_polygon_merged.gpkg \
  --merged-wfs-layer mansfield_polygons_in_buffers_merged \
  --council-gpkg /data/mansfield/spatial/base-map/mansfield_councils_land.gpkg \
  --council-layer cadastral_parcels \
  --uprn-gpkg /data/base-data/osopenuprn_202602.gpkg \
  --uprn-layer osopenuprn_address \
  --open-roads /data/base-data/oproad_gpkg_gb/Data/oproad_gb.gpkg \
  --theme-filter-regex "Land|Building" \
  --max-roi-rank 8 \
  --max-output-rank 200 \
  --output-prefix "$POLY_PREFIX"
```

主要输出：

```text
${POLY_PREFIX}_top200.csv
${POLY_PREFIX}_top20.csv
${POLY_PREFIX}_cases.csv
${POLY_PREFIX}.summary.json
```

`*_top200.csv` 是后面 polygon ranking 的输入。

### 3.3 对候选 polygon 排序

入口脚本：

```text
no_ai_candidate_boxes/polygon_ranking.py
```

Top5 推荐：

```bash
POLY_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_polygon_candidates
RANK_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_polygon_ranking_top5

/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/no_ai_candidate_boxes/polygon_ranking.py \
  --input-csv "${POLY_PREFIX}_top200.csv" \
  --output-prefix "$RANK_PREFIX" \
  --strategy relation_evidence_mix \
  --candidate-source-filter council_cadastral \
  --top-k 5 \
  --output-top-n 100 \
  --max-input-rank 200
```

如果想看 Top8：

```bash
POLY_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_polygon_candidates
RANK_PREFIX=/env/code/spatial-capture-monmouthshire/geocoding-box/tmp_results/my_run_polygon_ranking_top8

/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/no_ai_candidate_boxes/polygon_ranking.py \
  --input-csv "${POLY_PREFIX}_top200.csv" \
  --output-prefix "$RANK_PREFIX" \
  --strategy portfolio \
  --top-k 8 \
  --output-top-n 100 \
  --max-input-rank 200
```

polygon ranking 输出：

```text
<rank-prefix>_ranked_top100.csv
<rank-prefix>_selected_top5.csv
<rank-prefix>_case_summary.csv
<rank-prefix>.summary.json
```

如果 `--top-k 8`，则 selected 文件名会是：

```text
<rank-prefix>_selected_top8.csv
```

### 候选 polygon 准确度口径

我们当前关心的是相交，不关心 IoU：

```text
truth_intersects = candidate polygon intersects truth polygon
```

之前 Mansfield 检查大致是：

```text
Top5 polygon candidates: 约 85%-86%
Top8 左右收益较大
Top10 会继续涨一点，但候选数量也更多
```

## 4. 我经常会忘的文件名

### 单点 geocoder

```text
geocode_address_point.py
no_ai_candidate_boxes/address_point_geocoder.py
```

### 候选框

```text
run_no_ai_candidate_boxes.py
no_ai_candidate_boxes/pipeline.py
```

### 候选框 active stages

```text
no_ai_candidate_boxes/stages/build_input_candidates.py
no_ai_candidate_boxes/stages/ocr_candidate_rerank.py
no_ai_candidate_boxes/stages/openname_anchor.py
no_ai_candidate_boxes/stages/openroads_relation.py
no_ai_candidate_boxes/stages/plan_ocr_anchor_audit.py
no_ai_candidate_boxes/stages/multi_roi_candidates.py
no_ai_candidate_boxes/stages/corridor_augmented_roi.py
```

### 候选 polygon

```text
legacy_experiments/9_hybrid_polygon_rerank.py
no_ai_candidate_boxes/polygon_ranking.py
tools/build_mansfield_merged_basemap.py
```

## 5. 换 council 时改哪里

复制模板：

```text
configs/template_council.json
```

改这些：

```text
council_root
base_data_root
output_root
local_bbox
locality_names
allowed_locality_names
generic_place_names
paths.text_xlsx
paths.ocr_jsonl
paths.open_roads
paths.open_names
paths.open_names_dir
paths.full_gpkg
layers.full_layer
parameters.roi_side
parameters.evidence_roi_count
```

运行时指定：

```bash
/env/code/spatial-capture-monmouthshire/.venv/bin/python \
  /env/code/spatial-capture-monmouthshire/geocoding-box/run_no_ai_candidate_boxes.py \
  --config /path/to/my_council.json \
  --council-root /data/my-council \
  --truth-gpkg /path/to/input.gpkg \
  --truth-layer layer_name \
  --output-prefix /path/to/output_prefix
```
