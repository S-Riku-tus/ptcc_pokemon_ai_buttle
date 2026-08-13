# v28 検証結果

## 結論

`alakazam_ml_v28` は提出候補としての静的・回帰検証を通過しました。ユーザー指定の「ノコッチ前・ケーシィ後ろでは交代技を使わない」と「倒せるジュラルドンを同サイドのエースバーンより優先する」は、それぞれ決定論テストで確認済みです。

## 自動テスト

- `python -m pytest agents/alakazam/alakazam_ml_v28 -q`
- 164 passed
- v28固有回帰: 4 passed
- `validate_agent.py --agent alakazam_ml_v28`: passed
- デッキ: 60枚、22種、警告0

v28固有回帰で確認した局面:

1. ノコッチ前・ケーシィ後ろ: 交代技より`END`を選択。
2. ノコッチ前・攻撃可能なフーディン後ろ: フーディンを晒さず`END`を選択。
3. ケーシィ前・ノコッチ後ろ: テレポートでノコッチを前へ出す。
4. エースバーン前・ジュラルドン後ろがともに同ターンKO可能: Bossを使ってジュラルドンを選択。

## 成果物ハッシュ

| ファイル | SHA-256 |
|---|---|
| `fallback_policy.py` | `3fcdb252f207646cd6464e6164aeb76c6eb2c515303a39f4b7f5711af754913b` |
| `ranker_model.json` | `22f41bfa04b4224c566d74d2642f4d8703fa36448dd815cc9b45c61c759e0bbb` |
| `target_ranker_model.json` | `a58d27eabbfd638debf893cf584ec2d2a680f7953282b47331735bb39b9af5c4` |
| `deck.csv` | `57c7d4800cfc0f36581077a40b24912d33056cafcc14cca3783094ce6c122bfe` |
