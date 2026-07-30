# v33 変更点

- v32を複製し、v31/v32を変更せず保存
- Yushin train 35,704判断のepisode単位4-fold OOFを実装
- 6 base model、214特徴のcandidate selectorを追加
- validation採用ゲートを追加し、78.34%のselectorを自動不採用
- selector無効時にbase artifactを提出物から除外
- Run Away Drawの`backup_eta <= 1`停止を削除
- Run Away Drawの相手残りサイド1停止を削除
- 手札10枚停止を同ターン確定KO停止へ変更
- Enriching Energyの固定手札8枚停止を削除
- F/G再構築モードを追加
- Poffinの盤面幅上限をcore body 4から5へ拡張
- Froslass盤面でShayminを防御札扱いしない
- OOF compact-runtime一致検証を追加
- v33回帰テストを追加
