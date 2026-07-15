# Offline evaluation

Time test decisions: 1865

| Model | Semantic Top 1 | Top 3 | MRR | Weighted log loss |
|---|---:|---:|---:|---:|
| first_legal | 37.26% | 61.21% | 0.530 | 4.370 |
| action_frequency | 50.18% | 71.97% | 0.640 | 1.714 |
| handwritten | 48.64% | 74.85% | 0.640 | 6.212 |
| lightgbm_ranker | 74.17% | 90.59% | 0.833 | 1.094 |
| small_neural_ranker | 66.89% | 87.11% | 0.782 | 1.200 |

Exact Top 1: 72.40%; semantic Top 5: 96.14%; ECE: 0.222.

The ranker passes Gate 2. Boss, energy, Xerosic, and retreat remain outside the first model-controlled runtime scope because their per-type accuracy is weak.
