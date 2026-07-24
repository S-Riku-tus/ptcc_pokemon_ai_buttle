# alakazam_ml_v19

公開v18（submission `54926062`、取得時レート770.1、24勝25敗）を、
2026-07-24取得の現行上位フーディンログで再監査したChallengerです。

上位13提出の主流60枚はv18と完全一致していたため、デッキは変更していません。
変更対象は、公開ログとの乖離が再現できたTrainer評価だけです。

## 主な変更

- 《ナイトタイム鉱山》のカードIDを`1266`として正しく認識。
- Teraへの実効税、または展開済み相手のスタジアム上書き時だけ鉱山を使用。
- 鉱山の手札消費で現在のPowerful Hand KOを失う場合は使用禁止。
- 《スイレンのお世話》の合法な回収対象を数え、1/2/3枚で価値を分離。
- 3枚回収時の実際の手札差分`+2`を終端火力判定へ反映。

## 検証

- pytest: 88 passed
- 上位4提出の18,749判断への一致率: v18 59.4% -> v19 60.8%
- v19 vs v18: 518-482 / 1,000戦
- v19 vs v17: 262-238 / 500戦
- v19 vs v15: 202-198 / 400戦
- v19 vs オーロンゲv6: 262-138 / 400戦
- クラッシュ、不正手、policy/observation fallback: 0

同seedのオーロンゲ基準ではv18が285-115だったため、この対面はv18比で低下しています。
一方、公開上位の方策一致とv18ミラーは改善方向です。公開レート上昇は未証明であり、
提出後に対面分布を含めて判定する必要があります。

根拠は`ANALYSIS_V19.md`、変更一覧は`CHANGELOG_V19.md`、
検証詳細は`VALIDATION_REPORT_V19.md`にあります。

```powershell
.\.venv\Scripts\python.exe -m pytest agents\alakazam\alakazam_ml_v19 -q
.\.venv\Scripts\python.exe -X utf8 .\scripts\local_arena.py `
  alakazam_ml_v19 alakazam_ml_v18 --games 500 --seed 1901 --quiet
```
