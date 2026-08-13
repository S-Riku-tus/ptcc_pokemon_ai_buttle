# Validation report v19

## 静的・Golden-state

- pytest: 88 passed
- `fallback_policy.py` / `main.py` / 調査スクリプト: `py_compile`成功
- デッキ: 60枚、v18と同一
- ranker: v18と同一
- policy fallback / observation fallback: 0
- クラッシュ / 不正手: 0

追加した6状態を固定しています。

1. 役割のないNighttime Mineを拒否。
2. 現在支払えるTera攻撃を止めるNighttime Mineを許可。
3. 現在のPowerful Hand KOを失うNighttime Mineを拒否。
4. Lana's Aidの3枚回収を13,200点で評価。
5. Rule Boxポケモンと特殊エネルギーをLana対象から除外。
6. Lanaの1枚／2枚回収を別スコアで評価。

## 上位方策再生

| Agent | 対象 | 判断 | 一致 | 一致率 |
|---|---:|---:|---:|---:|
| v18 | 上位4提出261戦 | 18,749 | 11,138 | 59.4% |
| v19 | 同一観測 | 18,749 | 11,396 | 60.8% |

観測は時系列順に入力し、各対戦の先頭でagent memoryをresetしました。

## 判定

カードIDの不一致を解消し、上位方策一致を1.4ポイント改善し、
実行安全性を維持できました。強さは公開ラダーで確認する必要があります。

提出後は次を確認します。

- 60戦以上の勝率、Wilson 95%区間、レート
- Nighttime Mine使用回数と同ターン攻撃率
- Lana's Aidの回収枚数分布
- Archaludon、Grimmsnarl、Alakazam対面
- 1回も攻撃できない試合の割合
