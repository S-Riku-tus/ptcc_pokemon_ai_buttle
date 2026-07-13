# kashiwashira Team Rocket’s Spidops 再現版 v1

Kaggle「Pokémon TCG AI Battle」の1位代表提出 `54603674`（Rating 1255.2）について、公開リプレイ100戦から60枚と行動傾向を復元し、同じデッキを操るための新規ポリシーを実装したものです。

## 重要な範囲

- `deck.csv` は代表提出の60枚をそのまま再現しています。
- 元エージェントのソースコードはログに含まれていないため、`main.py` はコピーではありません。
- ロジックは、カード使用、進化、攻撃、エネルギーの移動元・移動先、Mewtwo exの追加打点選択、サーチ選択、勝敗別の盤面差を使ったクリーンルーム再構成です。
- 比較提出 `54613990` は、基本草エネルギー2枚を基本超エネルギー2枚へ変更した以外は同じ58枚です。本版はRatingの高い代表提出に合わせ、草9枚を採用しています。

## 提出に必要なファイル

- `main.py`
- `policy_base.py`
- `deck.csv`

`metadata.json`、`STRATEGY.md`、`LOG_ANALYSIS.md`、`VALIDATION.md`、分析CSV、テストは検証・保守用です。

## 検証

```bash
python -m py_compile main.py policy_base.py
python -m unittest discover -s tests -v
```

実対戦環境では、さらに以下を実行してください。

```bash
pytest
python validate_agent.py <agent-dir>
# 既存の対戦ハーネスで100戦以上
```

この作業環境にはプロジェクト本体の `vendor/cg` と対戦ハーネスがないため、実エンジン対戦までは行っていません。テストはAPI互換スタブによるインポート、60枚、提出ハンドシェイク、主要定数の検証です。
