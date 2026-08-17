# find_plot_in_plan

Given a scanned UK planning drawing, say **where on the page the application plot is** — as a
point and a bounding box.

One model call per page. No OpenCV, no post-processing: the coordinates are exactly what the
model returned, with out-of-range values flagged rather than corrected.

```bash
PY=/env/code/plot-reasoning/GeoPlanAgent/.venv/bin/python

$PY run.py --price                                    # pricing, no model calls
$PY run.py --src /path/to/panels --out results/       # whole folder, 10 workers
$PY run.py --src ... --limit 20 --stride 37           # deterministic sample
$PY run.py --src ... --no-metadata                    # image only, no case record
$PY run.py --src ... --render                         # also draw overlays
```

Results stream to `results/results.jsonl`, one line per page. Re-running skips pages already
present, so an interrupted batch resumes without paying twice.

## What it costs

| | |
|---|---|
| Model | `openai/gpt-5.6-terra`, $1.00 / $6.00 per Mtok |
| Measured tokens | 3130 in, 575 out per page |
| **Per page** | **$0.00658** |
| 783 pages (the Braintree wp7 panels) | $5.15 = **£4.09** |
| 1000 pages | $6.58 = **£5.22** |

Roughly half the bill is the image, half is the model's output. `--price` re-fetches live
rates and warns if they have moved.

Wall clock: a median 8 s per page from the model, so 10 workers put 783 pages at about ten
minutes. Serial would be nearly two hours — the runner is concurrent by default for that reason.

## The design decision that matters

**The prompt names no cues to look for.** It says what a plan of land is, says the plot is the
whole curtilage rather than the building footprint, lists what is *not* a site marking (north
arrows, scale bars, title blocks, stamps, lettering), and gives one rule: *if this is not a plan
of land, decline*.

Every earlier version enumerated cues — "a label with a leader line, hatching, a house number" —
and each list became a ceiling. The version with four cues **declined on 42.5% of pages**,
truthfully reporting that none of its four cues were present on drawings that mark the site some
other way. Removing the list entirely fixed that, and the model's free-text answers then named
marking kinds nobody had thought to enumerate: the applicant drawing *their* plot in full detail
while neighbours stay schematic; a plot marked by being the only footprint left as an open white
outline; whole sheets that are one plot bounded by a labelled hedge.

`prompt.txt` is checksummed (`load_prompt()` refuses to run if it changes) because every figure
below was measured against that exact text.

## The optional case record

`metadata.py` joins each panel to the council's own record — reference, address, proposal — from
`wp7_auto_address_date.csv`, matching on the panel stem's prefix (`93-00708-P_001-001_0001_p5`
→ `93-00708-P`). Sent by default; disable with `--no-metadata`.

Measured on 20 unseen pages, with the record versus without:

| | with | without |
|---|---|---|
| Located | **20/20** | 18/20 |
| Judged not a plan | 0 | 1 |
| Median point shift | 33 px | — |
| Shifted > 100 px | 1/20 | — |

The reasons became specific — the model started citing house numbers and property names from
the address. The small median shift reads as confirmation rather than redirection.

**One caveat to carry.** `not_a_plan` went from 1 to 0: on `88-01992-P_001-002_0004_p6` the
model called the page not a plan without the record and described a site with it. Whether a page
is a plan of land is a property of the image, and text about the application should not be able
to change that verdict. If a plan/not-plan gate matters to you, consider running that decision
without the record.

## What the numbers are, and are not

**There is no ground truth for this corpus.** No hand-marked site exists for any page. Nothing
here is an accuracy against truth. What was measured:

- **Against `gemini-3.1-pro` on 20 unseen pages** (a second opinion, not truth): the two models
  put the point in the same parcel on 9 of 18 comparable pages. pro scored 13/18 against itself,
  because on 5 pages its own point falls in no traced parcel — so 13 was the ceiling and terra
  reached 69% of it. A constant point that never sees the image scores 5/18, so terra is clearly
  reading the drawing; several cheaper models tested were not.
- **Reproducibility**: run twice on the same 20 pages, the `is_plan` and `found` verdicts were
  identical on all 20. The point moved a median 33 px. This model runs with reasoning enabled
  and therefore **rejects `temperature`** — it warns and ignores it — so answers vary slightly
  between runs by design.
- **Coordinate range**: 0 of 40 points fell outside the image. Other models tested did produce
  out-of-range coordinates, so the check stays.
- A human reviewed the output on 20 unseen pages and judged it the best of the thirteen
  configurations tried.

## Known limits

- **Composite sheets.** Some pages carry a location plan, a site plan and an elevation together.
  The model marks one location; there is no handling for "the same plot appears twice on this
  page".
- **Upstream leakage.** The panel folder is the output of an imperfect classifier: floor plans,
  elevations and title sheets are present. An audit put this at 7.6–16.7% of panels. `is_plan`
  catches many of them and reports `plan_kind` for the rest.
- **No boundary.** The output is a point and a box. Turning either into a traced polygon is out
  of scope here.
- **Colour is gone.** All 783 panels were checked: coloured-ink fraction is 0.00% on every one.
  A red line on the original paper is black in the scan, so no colour cue survives.

## Files

| | |
|---|---|
| `prompt.txt` | the instruction text, checksummed |
| `locate.py` | schema, one-page call, cost accounting |
| `metadata.py` | join to the council's case record |
| `run.py` | concurrent batch runner with resume |
| `render.py` | overlays of exactly what was returned |

Needs `OPENROUTER_API_KEY`, read from `GeoPlanAgent/.env`.
