# Satellite Grounding

Language-grounded object detection on VRSBench satellite imagery, using Grounding DINO.

## Setup
- `smoke_test.py` — confirms the model loads and MPS device works
- `build_samples.py` — builds `data/sample/samples.json` (15 image/query/box triples) from VRSBench validation annotations
- `run_baseline.py` — runs on 15 samples, computes IoU per image, prints Mean IoU / Acc@0.5

## Week 1: Baseline + cheap experiments

| Experiment | Mean IoU | Acc@0.5 |
|---|---|---|
| Baseline (grounding-dino-tiny, full queries, threshold 0.1) | **0.221** | **0.20** |
| Bigger model (grounding-dino-base) | 0.189 | 0.13 |
| Short queries (noun phrases, no position words) | 0.126 | 0.07 |
| Lower threshold (0.05) | 0.221 | 0.20 |

None of the three cheap fixes beat the baseline:
- **Bigger model** performed worse, model capacity is not the problem.
- **Short queries** performed worse — position/relation words ("eastern side," "near the terminal") turned out to carry real disambiguating signal, not just noise, especially for images with multiple similar objects.
- **Lower threshold** made no difference

**Working baseline: grounding-dino-tiny, full queries, threshold 0.1 (0.221 IoU / 0.20 Acc@0.5).** 
