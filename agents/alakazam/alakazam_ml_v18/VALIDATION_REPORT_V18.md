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
