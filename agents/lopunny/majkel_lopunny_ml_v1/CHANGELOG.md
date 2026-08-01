# Changelog

## 1.0.0 - 2026-08-02

- Majkel1337 submission 55137818 の386正常replayから全context教師模倣コーパスを構築。
- 教師席と60枚deck signatureを全試合で検証。
- LambdaRank候補順位モデルと可変選択数モデルを追加。
- LightGBM JSONを標準ライブラリのみで実行するランタイムを追加。
- 時系列train/validation/test、semantic/raw exact、Top-k、context別評価を追加。
- 既知ログ再生90.21%と未知時系列test 77.13%を分離して記録。
- 任意のrule baselineと安全ガード、単体テストを追加。
