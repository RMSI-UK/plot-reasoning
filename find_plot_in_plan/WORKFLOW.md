# Plan scan in, boundary and confidence out

One command. The pipeline is closed: an image goes in, a polygon and a calibrated probability
come out, and the probability has been validated on pages the pipeline had never seen.

```bash
python extract_boundary.py page.jpg                     # JSON to stdout
python extract_boundary.py page.jpg --overlay out.png   # and a picture to check it by
python extract_boundary.py --src panels/ --out results/ --workers 12
python extract_boundary.py --score-only results/        # re-score existing output, no API calls
```

## What comes back

```json
{
  "found": true,
  "vertices_page": [[418, 297], [512, 301], ...],
  "confidence": 0.87,
  "confidence_band": "high",
  "auto_usable": true,
  "flags": [],
  "locators_agreed": "3/3",
  "located_by": "openai/gpt-5.6-luna-pro",
  "what_it_traced": "the curtilage enclosing No. 18 and its rear garden ...",
  "usd": 0.019
}
```

`vertices_page` is in ORIGINAL page pixels, so it needs no transform to sit on the scan.

## The five stages

| | | |
|---|---|---|
| 1 | locate | three models on the whole sheet — luna, luna-pro, gemini-3.7-flash. Boxes clustered at IoU ≥ 0.5, largest cluster wins |
| 2 | window | a crop that CONTAINS the box, padded, up to 1024 px, never scaled below 1.0 |
| 3 | trace | gpt-5.6-terra on that crop; one retry allows a building outline when nothing else is drawn |
| 4 | check | seven checks, none needing an answer key |
| 5 | score | calibrated P(the ring is the right plot to within IoU 0.7) |

## What the confidence is worth

Held out — the model was fitted on 62 pages and these 97 are different pages, hand-drawn
**before** the pipeline ran, from a random sample it had never seen:

| band | pages | share | actually right | 95% CI |
|---|---|---|---|---|
| **high** | 24 | 25% | **88%** | 69–96 |
| medium | 27 | 28% | 67% | 48–81 |
| low | 36 | 37% | 22% | 12–38 |
| none (declined) | 10 | 10% | 90% | 60–98 |
| all | 97 | 100% | 58% | 48–67 |

Declining scores well because on a page with no boundary drawn, declining is the right answer.

**Use the high band and review the rest.** Over the 783-page corpus that is roughly 200 boundaries
you can ship at about 88%, for about £15 per 1000 pages.

## What it is not

Two pages in five are wrong at IoU 0.7 and about one in five is a completely different parcel.
This is a drafting tool with a trustworthy top lane, not an extractor. Overall accuracy resisted
three separate attacks, all documented in `geometry/optimise/index.html`:

- seeding a geometric line-network partition with the locate point — much worse (median IoU
  0.846 → 0.104)
- a subdivision second pass for over-reaching rings — one page, and its premise (that the error
  always runs one way) did not replicate blind
- asking a model to choose between the locators' own boxes — nothing, even though a correct box
  is among them 84% of the time

## Failure is loud

An exhausted account once cost a run 141 of 159 pages and wrote them out as ordinary declines,
producing a summary that read like a result. HTTP 401/402/403 now stops the run with exit code 2
and a message saying it is a billing problem. Completed rows stay in `boundaries.jsonl` and the
same command resumes from there.

## Re-fitting the confidence

`reliability_model.json` was fitted with `reliability.py`, on one page set and validated on
another — never the same pages, because two figures in this project came from thresholds chosen
after seeing the answer and both had to be withdrawn.

```bash
python reliability.py --fit snapv2 snapv2_noboundary --validate snapv2_blind100
```

The fit currently scores AUC 0.703 held out, which is slightly **worse** than counting review
flags alone (0.719). It is kept because it is calibrated and the flag count is not. One input has
never been tested inside it: the tracer's own confidence (AUC 0.674 on its own) was being dropped
by the pipeline and is now recorded as `trace_confidence`, but the fitting set predates that, so
its coefficient is currently zero. Re-run both sets and re-fit to close that gap.

## Ground truth

`labels/hand_labels.json` (62 pages) and `labels/hand_labels_blind.json` (100 pages, blind).
Serve the annotation tool with `python serve_handlabel.py` (port 8777) or
`HANDLABEL_SET=handlabel2 python serve_handlabel.py` (port 8778). Every edit is written to disk
atomically with the last 40 versions kept.

Quote the blind or combined figure, never the 62-page one: it is biased optimistic by about
8 points because it contains pages picked by eye for being hard.
