# cynthia_garchomp_v2（v1 + ex耐性対応）

v1との差分のみ記載。基本戦略は `agents/cynthia_garchomp_v1/STRATEGY.md` を参照。

## 背景

実ラダーep85178595でCrustle（「相手のex/Mega exポケモンのワザのダメージを受けない」）に敗北。
v1はこの耐性を知らず、Garchomp exで0ダメージのワザを計画し続けてデッキ残4まで消耗した。

## 変更点

- `EX_IMMUNE = {158, 207, 330, 345}`（Crustle等、wmh/ptcg-abcの実測リスト）
- `_atk_dmg`: Garchomp exのワザ（Corkscrew/Buster）は耐性持ちに**ダメージ0として計画**
  → 無駄なBuster（全エネ破棄）やボスの誤射がなくなる。スピリトムのRaging Curseは非exなので通る
- `gust_value`: ボスで耐性持ちを前に呼ぶ選択に-5000
- スピリトム主軸化は**条件付き**（相手の前が耐性持ち かつ ベンチのダメカン≥6〜8個）。
  無条件の主軸化・強制リトリートは実測で逆効果（45%まで悪化）だったため撤去済み

## 教訓（wmhの知見の再確認）

「大きな行動ゲートは逐点指標を悪化させる。局所的なスコア修正に留めよ」——
初版のスピリトム偏重（カウンター0でも前出し）は、テンポ損失＋70HPのサイド献上で
Crustle戦をむしろ悪化させた。条件を付けて局所化したら改善した。

## ベンチマーク（scripts/local_arena.py, エラー0）

- **vs Crustle: 77%（30戦。v1は70%）**
- vs cynthia_garchomp_v1ミラー: 50%（60戦）— 非耐性対面では挙動保存を確認
- vs Grimmsnarl: 85% / vs Kangaskhan: 100%（回帰なし）
