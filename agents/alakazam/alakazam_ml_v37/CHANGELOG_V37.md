# v37 changelog

- Majkel1337 999episode、Yushin Ito 176episode、Rmy 699episodeを追加教師候補として抽出
- 追加合計93,050判断、822,885意味候補を監査
- 直接追学習とmulti-teacher consensus memoryを検証し、不採用
- Rmyログ専用action-type expertを学習
- v36 primary 85%＋Rmy expert 15%の確率blendを実装
- action-type gate thresholdを0.50から0.45へ変更
- `rmy_type_model.json`をpure-Python compact runtimeへ追加
- v37回帰テスト3件を追加
- test Top-1を81.62%から81.78%へ改善
