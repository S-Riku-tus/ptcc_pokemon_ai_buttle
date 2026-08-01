# alakazam_ml_v37

v37はv36のrich action-type modelを主軸に、上位Rmyの699試合から学習した独立type expertを15%だけblendするmulti-teacher challengerです。

- primary: v36 Yushin rich type model 85%
- expert: Rmy type model 15%
- type gate threshold: 0.45
- 同一type内: v34 LambdaRank
- v31系安全ゲート、teacher memory、v29 fallbackを維持
- runtimeはPython標準ライブラリのみ
- デッキ60枚はv31～v36と同一

時系列test Top-1は81.78%（7,236 / 8,848）で、v36から14判断改善しました。目標90%は未達です。v31をchampionとして維持し、v37はラダー検証用challengerとします。
