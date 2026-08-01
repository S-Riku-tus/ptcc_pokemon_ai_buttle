# alakazam_gpt_v4

> **軸について**: `alakazam_gpt_*`はChatGPTアプリ側で作成した外部由来の系統です。リポジトリ内の`alakazam_ml_*`とは別軸として管理し、バージョンはv1から独立に採番しています（旧`alakazam_ml_v37`＝本`alakazam_gpt_v4`）。本文中の`alakazam_ml_v29/v31`は継承元である別軸エージェントを指します。

v4はv3のrich action-type modelを主軸に、上位Rmyの699試合から学習した独立type expertを15%だけblendするmulti-teacher challengerです。

- primary: v3 Yushin rich type model 85%
- expert: Rmy type model 15%
- type gate threshold: 0.45
- 同一type内: v1 LambdaRank
- `alakazam_ml_v31`系安全ゲート、teacher memory、`alakazam_ml_v29` fallbackを維持
- runtimeはPython標準ライブラリのみ
- デッキ60枚は`alakazam_ml_v31`およびv1〜v3と同一

時系列test Top-1は81.78%（7,236 / 8,848）で、v3から14判断改善しました。目標90%は未達です。`alakazam_ml_v31`をchampionとして維持し、v4はラダー検証用challengerとします。
