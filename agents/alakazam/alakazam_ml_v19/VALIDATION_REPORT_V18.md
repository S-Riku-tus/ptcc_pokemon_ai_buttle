# Validation report v18

## 静的・Golden-state

- pytest: 82 passed
- `fallback_policy.py` / `ml_runtime.py` / `main.py`: `py_compile`成功
- デッキ: 60枚、v17とbyte単位で同一
- ranker: v17とbyte単位で同一
- policy fallback / observation fallback: 0
- クラッシュ / 不正手: 0

Golden-stateは次を固定しています。

1. Hammer後のHP340 Active KOがRiolu Bossを上回る。
2. リッチ＋進化後のHP340 Active KOがHariyama Bossを上回る。
3. Dawn後のHP440勝利KOがMakuhita Bossを上回る。
4. Activeを倒せず、Boss対象で勝てる場合はBossを維持。
5. 相手ベンチなしの確定KOでは即攻撃。
6. Activeノココッチから盤面全滅へ直結する特性を優先。
7. 特性ロックとスコア辞退を診断上で区別。
8. 1体のAbraをRare CandyとKadabra進化に重複利用しない。
9. Abraが2体ならRare CandyとKadabra進化を合法的に両立する。
10. Hammer後も非エネルギー由来の効果耐性が残る場合はKO扱いしない。

## ローカル対戦

| 対戦 | 結果 | 勝率 | Wilson 95% CI |
|---|---:|---:|---:|
| v18 vs v17 | 505-495 / 1,000 | 50.5% | 47.4-53.6% |
| v18 vs v15 | 203-197 / 400 | 50.7% | 45.9-55.6% |
| v18 vs オーロンゲv6 | 247-153 / 400 | 61.8% | 56.9-66.4% |

別seedの200戦smokeではv17へ105-95、最終hardening後の200戦では99-101でした。
ローカル検証の合計は2,200戦です。全試行でクラッシュ、不正手、fallbackは0でした。

## 解釈

v17・v15との信頼区間は50%を含みます。したがって、ローカル勝率の向上を
主張する結果ではありません。一方、クラッシュや不正手なしで既存性能を維持し、
対象の3ルートをGolden-stateで確実に反転できています。

公開ラダーでは対戦相手分布が異なるため、次の判定はsubmission後に行います。

- Boss後の取得サイド分布
- `boss_reachable_active_ko_blocks`の発火局面
- 初攻撃後の非攻撃ターン
- 公開60戦以上の勝率とWilson区間
- レート833.1およびv15の907.5との比較
