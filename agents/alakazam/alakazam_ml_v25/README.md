# alakazam_ml_v25

v24の実戦60戦と、リーダーボード上位Yushin Itoの1000戦を比較して作成した
高ラダー向けAlakazam/Dudunsparceエージェントです。

v24のデッキ、行動モデル、対象ランカーを維持し、次の三点だけを狭く修正します。

- Grimmsnarl/Froslass盤面では継続ダメカン源のFroslassを優先して止める
- KO後は攻撃系列の時間価値をDunsparce盾より優先する
- AlakazamミラーではFlip the Script回収権のないFezandipiti exを露出しない

詳細:

- `ANALYSIS_V25.md`: データ、原因、設計判断
- `CHANGELOG_V25.md`: 実装差分
- `VALIDATION_REPORT_V25.md`: 回帰、リプレイ、ローカル対戦の結果

新しいラダーレートは未計測です。teacher-forced replayとローカル対戦は
発火範囲と安全性の検査であり、提出後の勝率を保証するものではありません。
