# v25 変更点

v24を親とし、デッキと二つの学習モデルを固定したまま、実ログで確認できた
三つの高ラダー判断差だけを修正しました。

## Runtime

- オーロンゲ/Froslass/Munkidori盤面で、Froslassが残っている間は
  継続ダメカン源をBoss対象として優先
- Froslassがいない時は、エネ付きMunkidoriを従来どおり役割対象として評価
- KO後の昇格で、未完成でもAlakazam系列の時間価値をDunsparce盾より上に評価
- 進化橋がない裸のAbraはDunsparceより下に残し、事故時の盾を維持
- 自分のActiveが倒された後の一ターンだけFlip the Script回収窓を記憶
- Alakazamミラーでは、回収窓がなく場に別の体がいる時のFezandipiti ex展開を抑止
- Powerful Handのダメカン計算をFezandipiti exの公開KO危険判定へ追加

## Diagnostics

- `v25_froslass_priority_scores`
- `v25_mirror_fez_blocks`
- `v25_mirror_fez_recovery_windows`
- `v25_line_promotion_scores`

## Preserved

- deck.csv
- v20 action ranker
- v24 Yushin-1000 target ranker
- v24までの全決定ゲートとテスト（v25で期待値を更新した昇格テストを除く）
