# v19変更一覧

## P0: Nighttime Mine

- `C.BATTLE_CAGE = 1264`を`C.NIGHTTIME_MINE = 1266`へ修正。
- `_opp_has_tera()`を追加。
- `_nighttime_mine_tax_stops_active()`を追加。
- `_nighttime_mine_worthwhile()`を追加。
- Tera税、相手スタジアム上書き、KO火力維持、山札残数を明示的に判定。
- 未知Trainer既定値9,000による毎試合の自動使用を廃止。

## P1: Lana's Aid

- `_lana_recoverable_count()`を追加。
- ルールを持たないポケモンと基本超エネルギーだけを回収対象として数える。
- 対象0枚では使用禁止。
- 対象1/2/3枚以上を4,500/10,500/13,200点に分離。
- 低山札かつ3枚以上では16,000点。
- `_hand_delta()`へ実回収数を反映し、3枚回収を純増`+2`として扱う。

## 変更しなかったもの

- 60枚のデッキ
- rankerモデル、特徴量、閾値、ML適用範囲
- v18の確定ルート探索とBoss比較
- 攻撃不能、一時無敵、Mist Energy、Articuno、Fezandipiti処理
- 広い後続フーディン再建規則

## 調査ツール

- `scripts/analyze_alakazam_v19_top40.py`
- `scripts/analyze_alakazam_policy_imitation.py`
