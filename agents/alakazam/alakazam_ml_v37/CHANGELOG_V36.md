# v36 変更点

- v35を複製し、v31〜v35を変更せず保存
- 行動種ごとのrich candidate-set集約を追加
- runtime表現を356値から1,052値へ拡張
- validationで追加80特徴を選び、最終入力を356特徴へ圧縮
- リーサル維持、手札増減、ドロー、進化ルート不足、対象・候補順序を行動種別に集約
- 2,497木のXGBoost行動種モデルを純Python JSONへ変換
- XGBoostと同じfloat32 split比較を実装
- confidence 0.50のhard type gateを維持
- test Top-1を81.60%から81.62%へ更新
- 目標90%未達と、v35との差が実質的に同等であることを明記
- v36 runtime契約テストを3件追加
