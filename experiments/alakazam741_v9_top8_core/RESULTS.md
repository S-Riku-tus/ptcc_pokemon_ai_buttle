# RESULTS — alakazam741_v9_top8_core

## 静的検証

- `python -m py_compile` main.py / policy_base.py … OK
- `validate_agent --agent alakazam741_v9_top8_core` … deck_size 60, warnings なし, Validation passed
- `pytest tests/test_alakazam741_v9_top8_core.py` … **23 passed**（要求22項目＋ミラー補助1）
- `pytest`（全体）… **89 passed**

## 対戦結果（`local_arena`, 席交換均等 100/100, 200戦/シード）

| 相手 | 勝率(除ドロー) | 備考 |
|---|---|---|
| v8 | seed0 56.5% / seed1 55.0% / seed2 54.5% / seed3 61.0% → **4シード集計 446/800 = 55.75%** | 目標 ≥55% **達成** |
| v3 | seed0 43.0% / seed1 42.0% / seed2 42.0% | 目標 ~50% **未達**（下記診断） |
| v2 | seed0 54.5% / seed1 54.0% ≈ **54%** | 参考目標クリア |

### v3 未達の診断（対面別分岐は追加しない）

v3 は系列全体に強い。同一条件（seed2, 200戦）で **v8 vs v3 = 40.5%、v2 vs v3 = 36.0%、v9 vs v3 = 42.0%**。
つまり v9 は v3 相手でも系列中で最も勝てており、これは v9 の退行ではなく **v3 の絶対的な強さ**。初攻撃・攻撃率・退避は目標内（下表）で、差の主因は「v9固有の弱点」ではなく、v3 の総合力である。行動指標側の唯一の弱点はノココッチ循環回数が目安下限（2.5）を下回る点と、山札切れ疑いが v8 より多い点（テンポ重視 Yushin 型の副作用）。

## 行動指標（v9 as primary, vs v8, 200戦）

| 指標 | 実測 | 目標 | 判定 |
|---|---|---|---|
| 平均初攻撃（自ターン） | 2.12 | ≤2.3 | ✅ |
| 2回目自ターンまで攻撃率 | 68.5% | ≥65% | ✅ |
| 全自ターン攻撃率 | 77.7% | ≥72% | ✅ |
| 攻撃回数/試合 | 5.17 | — | — |
| フーディン攻撃回数/試合 | 4.02 | — | — |
| ノココッチ特性/試合 | 1.85 | 2.5–4.5(目安) | ⚠ 下限未満 |
| ノココッチ特性ターンの攻撃率 | 94.1% | 高いほど良 | ✅ |
| 退避/試合 | 0.21 | ≤0.3 | ✅ |
| 退避後の同ターン攻撃率 | 100% | ≥90% | ✅ |
| 攻撃可能なのにEND | 0 | 0 | ✅ |
| 0ダメージPowerful Hand | 0 | 0 | ✅ |

## 安全（全600+戦を通じて）

クラッシュ 0 / 違法選択 0 / policy fallback 0 / observation fallback 0 / 最後の1体ノココッチ自滅 0 / 0ダメージPowerful Hand 0 / 攻撃可能END 0。現在KOを手札消費で失う件はゴールデンテストで担保（`_preserves_attack`）。

山札切れ疑い（負け かつ 山札0到達の近似指標）は 200戦あたり 8–18 件で、v8 の約0件より多い。これはテンポ最大化（Yushin型）の既知トレードオフで、`turns_to_win` を実ゲーム長（サイド/0.7）へ補正して 22→8–18 に低減済み。さらなる低減は未解決課題（下記）。

## 実行コマンド

```
python -m py_compile agents/alakazam741_v9_top8_core/main.py agents/alakazam741_v9_top8_core/policy_base.py
python scripts/validate_agent.py --agent alakazam741_v9_top8_core
python -m pytest tests/test_alakazam741_v9_top8_core.py
python -m pytest
python scripts/local_arena.py alakazam741_v9_top8_core alakazam741_v8 --games 200
python scripts/local_arena.py alakazam741_v9_top8_core alakazam741_v3 --games 200
python scripts/local_arena.py alakazam741_v9_top8_core alakazam741_v2 --games 200
python scripts/analyze_alakazam_policy_metrics.py alakazam741_v9_top8_core alakazam741_v8 --games 200 --out experiments/alakazam741_v9_top8_core
```

## 未解決課題

1. **v3 に対し ~42%**: v3 は系列全体に強く、対面別分岐なしでの改善は将来の一般ロジック（例: ミラーでの攻撃継続をさらに徹底、実効山札の精緻化）に委ねる。
2. **ノココッチ循環が目安下限未満（1.85 < 2.5）**: 動的山札ゲートがやや保守的。山札安全を保ちつつ循環を増やす調整が次の候補（ただし山札切れとのトレードオフ）。
3. **山札切れ疑いが v8 より多い**: `turns_to_win` の係数や実効山札の見積り精緻化で改善余地。
4. Kaggle Rating 1000+ はローカル代理指標では判定不可（下記報告参照）。
