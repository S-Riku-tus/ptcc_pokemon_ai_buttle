# alakazam741_v7.2 修正報告

## デッキ差分

- Crushing Hammer 1120: 2 → 0
- Battle Cage 1264: 2 → 3
- Night Stretcher 1097: 2 → 3
- Boss's Orders 1182: 0を維持
- 合計60枚

## ロジック変更

- クラッシュハンマー定数、使用条件、Tier分岐、相手エネルギー選択処理、テストを削除
- バトルケージを、相手スタジアムまたはベンチ攻撃脅威がある場合だけ使用
- 必要なバトルケージは`PRESSURE`/`ENDGAME`で攻撃前へ昇格
- 夜のタンカを、単なるトラッシュ存在判定から進化ルート判定へ変更
- 夜のタンカの対象選択を、ケーシィ・ユンゲラー・フーディン・ノコッチ系・基本超エネルギーごとに評価
- 有効な夜のタンカは解決後の手札枚数を0変化として致死維持判定
- 攻撃可能なフーディンに後続がない時だけ、ケーシィ・なかよしポフィン・夜のタンカを攻撃前へ昇格
- すでに後続がある場合は追加展開を攻撃より下へ維持
- 現在のKOを失うバトルケージやケーシィ展開は引き続きブロック
- 未使用importと変数を整理

## 維持した安全条件

- 最後の1体のActiveノココッチによるRun Away Drawを無条件禁止
- Activeノココッチは、ベンチに攻撃可能なフーディンがいる場合だけ退避
- 低山札時のACTIVATE辞退
- 勝利KOの即攻撃
- 手札消費による致死喪失の防止
- 攻撃可能時のEND禁止
- 改造ハンマーを効果防止解除後の即KOに限定
- リーリエの前に確定盤面行動を優先
- ハイパーアロマの3枚セット選択
- 盤面全体の効果防止判定
- BasePolicy、PrizeTracker、make_agent、合法フォールバック

## 検証結果

- `main.py` / `policy_base.py` / テストのPython構文チェック成功
- デッキ枚数60枚
- Boss's Orders 0枚
- Crushing Hammer 0枚
- Battle Cage 3枚
- Night Stretcher 3枚
- 互換APIスタブ上のGolden-stateテスト32件成功
- `main.py`: 1186行
- `main.py`の関数数: 74

## テスト上の制約

この実行環境には実リポジトリの`vendor/cg`がなかったため、カード・Observation APIの互換スタブを用いてGolden-stateテストを実行しました。構文、選択ロジック、優先順位、デッキ構成は確認済みです。最終提出前には実リポジトリで次を実行してください。

```powershell
python .\scripts\validate_agent.py --agent alakazam741_v7
python -m pytest tests/test_alakazam741_v7.py
python .\scripts\build_submission.py --agent alakazam741_v7
```
