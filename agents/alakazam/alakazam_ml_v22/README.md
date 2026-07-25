# alakazam_ml_v22

v21の60枚・学習済みLightGBM・ML担当範囲を固定したまま、上位フーディンログとの比較で判明した**攻撃継続と循環の差**をルール側へ反映したChallengerです。

## v22の基本思想

v21は、現在のKOが成立するとノココッチを早く止めるため、山札切れは減りました。一方、上位ログでは勝利試合でもノココッチを約3回使い、12～13枚程度の必要十分な手札から、現在の攻撃と次の攻撃を並行して作っています。

v22では、単にケーシィやユンゲラーが見えているだけでは後続完成としません。

```text
安全な後続 = backup_eta <= 1
```

を中心条件にし、現在のKO後に次の攻撃役が間に合わない場合は、KOが成立済みでもノココッチを使います。

## 主な変更

- **後続判定**：場に進化ラインがあるだけでなく、エネルギーを含め1ターン以内に攻撃可能かを評価
- **ノココッチ循環**：現在のKOが勝利確定でなく、`backup_eta > 1`なら循環を継続
- **必要十分な手札**：13枚をソフト目標にし、現在の攻撃と後続が完成した後の任意ドローを抑制
- **ノコッチ再展開**：循環後にエンジンが消え、後続が遠い場合は追加ケーシィよりノコッチを優先
- **キチキギスex**：明確な100ダメージ対象、効果ロック解除、ロケット団のフリーザー対策がある場合だけ段階的に育成
- **キチキギス昇格**：3エネルギー付きでベンチ100ダメージKOがある場合、攻撃可能な昇格として認識

## 変更していないもの

- `deck.csv`の60枚
- `ranker_model.json`
- 274特徴量・50木の蒸留LightGBM
- confidence threshold 0.37
- MLのライブ担当範囲
- Boss、Hammer、クセロシキ、効果無効、退避、気絶後昇格の既存安全処理
- 山札フロア、最後の1体ノココッチ禁止、確定KO維持ゲート

## 配置

```text
agents/
└─ alakazam_ml_v22/
   ├─ main.py
   ├─ fallback_policy.py
   ├─ policy_base.py
   ├─ ml_runtime.py
   ├─ ml_features.py
   ├─ common_runtime.py
   ├─ ranker_model.json
   ├─ deck.csv
   └─ metadata.json
```

## 検証

- Python compile：成功
- pytest：128件成功
- v22固有Golden-state：9件成功
- deck：60枚
- ACE SPEC：1枚
- v21とdeck/modelのSHA-256一致

公式`cg`と実対戦ハーネスがこの環境にないため、Validation Episodeと対戦Ratingは未検証です。詳細は`ANALYSIS_V22.md`と`VALIDATION_REPORT_V22.md`を参照してください。
