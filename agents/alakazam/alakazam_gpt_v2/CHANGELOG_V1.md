# v1 変更点

- v32を基盤にv1を新規作成し、v31・v32を変更していない
- Yushin上位ログを生JSONから再構成
- 49,590判断・443,714意味候補を従来定義と一致させた
- 640盤面／候補特徴へ34の方策状態特徴を追加
- 時系列未学習test Top-1を78.71%から80.63%へ改善
- Top-2包含93.30%、Top-3包含96.99%を確認
- OOF residual、Top-2 selector、GRU selectorを検証し、提出本体では不採用
- 漏洩を含んだ初期sequence実験を検出・破棄
- rankerを312木・674特徴の標準ライブラリJSONへ蒸留
- runtimeへepisode内sequence stateを実装
- v1/v32/v31環境変数の後方互換を維持
- v31/v32のデッキ、安全ゲート、v29 residual、teacher memoryを維持
- v1追加・継承テスト181件成功
