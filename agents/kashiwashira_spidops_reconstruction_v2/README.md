# kashiwashira Team Rocket’s Spidops 再構成版 v2

Kaggle「Pokémon TCG AI Battle」の1位代表提出 `54603674` の60枚を維持し、初回再構成版 `54655021`（約530 Rating）の43公開戦を、1位版の99公開戦と比較してロジックを修正した版です。

## v2で直した中心問題

- Poké Padの候補をサポーターとして評価していたため、28回中28回が空選択になっていたバグを修正
- Poké Padをサポーター使用後に禁止していた誤判定を削除
- Factoryを手札7枚以上で停止していた条件を削除し、山札2枚以下だけ停止
- 非KO攻撃より先に、安全な検索・ドロー・進化・加速を処理する`PRE_ATTACK`優先帯を追加
- 確定KO・勝利KOは引き続き展開より上位
- TransceiverをProton偏重から、初動Proton／展開後Ariana中心へ変更
- Giovanniを同ターンKOが成立する場合だけ使用
- 初期ActiveをMimikyu優先へ変更
- Mimikyuから準備済みSpidopsへのピボットを追加
- Crustle、Alakazam、Mega Lucario ex、高速exへの限定的な対面優先を追加

## 提出に必要なファイル

- `main.py`
- `policy_base.py`
- `deck.csv`

## ローカル検証

```bash
python -m py_compile main.py policy_base.py
python -m unittest discover -s tests -v
```

この作業環境ではAPI互換スタブによる26テスト、構文検査、ZIP整合性検査まで実施しています。実プロジェクトの`vendor/cg`と対戦ハーネスは存在しないため、実エンジン対戦は未実施です。

詳細は次を参照してください。

- `V1_FAILURE_ANALYSIS.md`：530 Rating版と1位版の比較
- `STRATEGY.md`：v2の行動順序と対面方針
- `VALIDATION.md`：実施済み検証と提出前チェック
- `CHANGELOG.md`：全修正点
