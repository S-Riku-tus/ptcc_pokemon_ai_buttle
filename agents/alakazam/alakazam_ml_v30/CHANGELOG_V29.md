# v29 changelog

## Added

- 上位5チーム361試合から学習する順位重み付きLambdaRank trainer
- 重複カードcopyを同じ正解として扱うsemantic teacher label
- 自分・相手の順序付きベンチslot特徴
- 合法手集合、候補数、同カード候補数、行動種別候補数の特徴
- v28方策と旧ランカーのscore、gap、rankを使うresidual features
- episode hash単位のtrain/validation/test分割
- exact、semantic、intent、top-k、順位重み付き一致率の評価script
- runtime診断とv29専用回帰test

## Changed

- MAIN contextの選択を限定的な旧ML overrideから、全合法手を比較するteacher rankerへ変更
- default confidence thresholdを0.20へ変更
- v28の旧rankerを `legacy_ranker_model.json` として保存
- 低信頼・対象外contextはv28 deterministic policyへfallback

## Safety

- 即時lethal、Boss確定route、現在のKOを保護
- 攻撃可能なActiveフーディンでのENDを禁止
- 2体以下の盤面でノココッチを消す特性を禁止
- 残り時間、モデル読込、合法性のfallbackを維持

## Not changed

- 60枚のデッキ
- v28 deterministic fallback policy
- v28 target ranker
- nested selectionの制御
