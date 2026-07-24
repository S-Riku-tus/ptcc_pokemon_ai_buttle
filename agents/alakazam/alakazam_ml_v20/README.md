# alakazam_ml_v20

`alakazam_ml_v19`を親に、Powerful Handの手札火力とサイドレースを同じ
ターゲット計画へ統合したChallengerです。60枚デッキ、LightGBMモデル、
ML特徴量、閾値`0.37`は変更していません。

## 主な変更

- 残りサイドが2枚または3枚で、Boss使用後の手札でもベンチの対象を倒して
  勝てる場合、《ボスの指令》をMAINの絶対条件として選択。
- 相手の全ポケモンをプライズ、残りサイド、役割、進化・エネルギー投資、
  Activeを倒す際の省資源性から順位付け。
- v18/v19のActive専用bitmask探索を任意ターゲットへ一般化。最優先対象が
  到達不能なら次点を試す。
- ベンチKOではBossの手札`-1`とSupporter枠を予約し、防止Energyには
  Enhanced Hammerを対象個体へ結び付けてからBossを実行。
- Powerful Handの必要手札を`ceil(残HP / 20)`で算出。確定KOルート到達後も
  5枚以上の余剰がある場合だけ、ベンチのRun Away Drawなど任意ドローを抑制。
- Dawnの実手札差分を`+2`へ統一し、同コストのKOルートは過剰ダメージが
  小さい方を選択。

Xerosic対策として余分な手札バッファは設けていません。何枚持っていても
3枚へ減らされるため、現在のKOを優先し、不要なデッキ消費を残す方針です。

## 検証

- pytest: 109 passed
- v20 vs v19（最終構成、seed 1901）: 245-255 / 500戦
- 勝ち確Boss: 18機会中18回実行
- 上位4提出の18,749判断への一致率: v19 60.782% / v20 60.553%
- クラッシュ、不正手、policy/observation fallback: 0

一致率の`-0.229pt`は、上位ログでも見送られていた勝ち確Bossと、
ターゲット到達後の過剰ドローを意図的に変更した結果です。ローカルmirrorは
互角範囲であり、公開ラダー上昇は未証明です。

診断用ロールバックとして`ALAKAZAM_V20_TARGET_ROUTES=0`と
`ALAKAZAM_V20_HAND_GATE=0`を用意しています。既定値はいずれも有効です。
余剰閾値は`ALAKAZAM_V20_HAND_SURPLUS`（既定`5`）で診断できます。

```powershell
.\.venv\Scripts\python.exe -m pytest agents\alakazam\alakazam_ml_v20 -q
.\.venv\Scripts\python.exe -X utf8 .\scripts\local_arena.py `
  alakazam_ml_v20 alakazam_ml_v19 --games 500 --seed 1901 --quiet --diag-json
```

設計根拠は`data/analysis/v20_second_ai_analysis.md`、変更一覧は
`CHANGELOG_V20.md`、検証詳細は`VALIDATION_REPORT_V20.md`にあります。
