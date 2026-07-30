# v33 検証報告

## OOF分離

- 教師: Yushin Ito / submission 54773249
- train: 35,704判断
- fold: episode単位4分割
- base model: 6
- selector特徴: 214
- OOF欠損: 0
- 構成選択: validationのみ
- test確認: 構成と採否確定後に1回

| 構成 | validation | test | 採用 |
|---|---:|---:|---|
| v32 recency単木 | 77.97% | 78.71% | 継続 |
| v32研究6モデルblend | 80.21% | 80.12% | 研究参照 |
| v33 OOF selector | 78.34% | 79.18% | いいえ |
| v33 OOF base oracle | 83.25% | 84.62% | 上限 |

採用閾値は80.41%です。v33 OOF selectorは未達のため`selector_model.json`の`enabled`をfalseにしました。提出アーカイブ作成時は`selector_base_*.json`を自動除外します。

## 実装検証

- v33ディレクトリ回帰: 185成功 / 0失敗
- 静的`validate_agent.py`: 成功
- デッキ: 60枚 / 22種類 / 警告0
- compact-runtime一致: 25判断、最大絶対スコア誤差0.0
- v31対v33: 40戦、20-17-3でv31
- v31対Grimmsnarl v7: 20戦、13-7
- v33対Grimmsnarl v7: 20戦、13-7
- クラッシュ: 0
- 違法手: 0

## 昇格判断

v31をchampionとして維持します。v33はselector昇格版ではなく、Run Away Draw/F/G/Poffin/Shayminの実戦A/B用challengerです。

90%は未達です。また今回のtestは開発用として既知なので、完全未閲覧holdoutの代替にはなりません。
