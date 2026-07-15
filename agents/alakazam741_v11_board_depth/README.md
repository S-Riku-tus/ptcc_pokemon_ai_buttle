# alakazam741_v11_board_depth v11.1

v10の攻撃テンポを維持しながら、後攻展開・盤面全滅・後続攻撃停止を改善する候補版。Validation Episodeを失敗させたACE SPEC重複を修正済み。

## 提出対象

- `main.py`
- `policy_base.py`
- `deck.csv`

## ACE SPEC

ハイパーアロマ1枚だけを採用する。エンリッチエネルギーは不採用で、エネルギーは基本超3枚＋テレパスサイコ4枚。

## ローカル検証

```bash
python test_v11_board_depth.py
```

25件のGolden-stateテスト、構文検査、デッキ規則検査を実行する。
