# CHANGELOG V13

## Deck

- `Max Rod` 1枚を削除
- `Enriching Energy / リッチエネルギー` 1枚を追加
- それ以外の59枚はv11と同一

## Deterministic policy

1. Rich Energyの4枚ドローを手札純増+3として評価
2. ノコッチ／ノココッチへの即時・将来循環を最優先
3. 現在KO＋後続ETA 1以下なら過剰ドローを停止
4. Bossで現在または同ターン中に確実に到達する高サイドActive KOを、低サイドBench KOへ置き換えない
5. Mist・効果保護対象へのPowerful Handを禁止
6. Team Rocket's Articunoの保護範囲を正しく限定
7. Articunoロック時のBoss脱出とFezandipiti ex別攻撃経路
8. 公開されたMist個体数と対面シグネチャに基づくHammer温存
9. v12のTeleport差分は不採用

## ML runtime

- モデルは変更なし
- Rich/Mist関連の将来学習用特徴を追加
- Enriching EnergyをPsychic Energyとして扱う誤りを修正
