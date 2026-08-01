# Changelog

## 2.0.0 - 2026-08-02

- 未知時系列test 85%を新しい主目標に設定。
- 行動種gate、canonical memory、8 ranker sweep、ensemble/oracle、pairwise、DeepSetsをvalidation限定で比較。
- 全合法手集合をmean/max poolingする75,809 parameter DeepSets residualを採用。
- DeepSetsを標準ライブラリだけで評価する純Python runtimeを追加。
- v1 test 77.13%からv2 77.48%へ非強制semantic exactを改善。
- 既知ログruntime replay 90.44%、legal 100%、p95 108.07msを確認。
- 85%未達を明記し、agent状態をoffline candidateに維持。
