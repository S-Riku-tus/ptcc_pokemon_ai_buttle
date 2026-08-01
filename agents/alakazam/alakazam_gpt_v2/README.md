# alakazam_gpt_v2

> **軸について**: `alakazam_gpt_*`はChatGPTアプリ側で作成した外部由来の系統です。リポジトリ内の`alakazam_ml_*`とは別軸として管理し、バージョンはv1から独立に採番しています（旧`alakazam_ml_v35`＝本`alakazam_gpt_v2`）。本文中の`alakazam_ml_v29/v31`は継承元である別軸エージェントを指します。

v2はv1を基盤に、行動種分類→種内候補順位付けの階層型方策を追加したchallengerです。

時系列分離test Top-1は81.60%で、v1の80.63%を上回りました。行動種分類は84.82%、正しい行動種が既知なら候補選択は95.59%です。目標90%は未達です。

提出runtimeは標準ライブラリのみで動きます。v1 ranker、`alakazam_ml_v29` fallback、決定論安全ゲート、teacher memoryを維持しています。
