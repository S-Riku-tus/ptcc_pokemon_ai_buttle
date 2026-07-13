# 検証結果と実リポジトリでの次手

## この環境で完了した検証

- 代表提出 `54603674` の `deck.csv` を全リプレイから照合し、60枚・19カード種を固定。
- 比較提出 `54613990` は、基本草2枚を基本超2枚へ置換した以外の58枚が同一であることを確認。
- 100代表リプレイと85比較リプレイについて、分析スクリプトを再実行し、同梱JSON/CSVを再生成。
- `main.py`、`policy_base.py`、分析スクリプト、テストのPython構文検査に成功。
- API互換スタブ上の10テストに成功。
- Mewtwo exの160/220/280、Brave Bangleの+30、草9枚のデッキ、先攻選択、合法フォールバックをテスト。

## この環境では未実施

プロジェクト本体の実 `vendor/cg`、`validate_agent.py`、ローカル対戦ハーネスがこの作業用コンテナにはないため、実エンジンでの対戦、Rating再現、Kaggle提出は未実施です。元エージェントのソースコードもZIPには含まれていないため、ロジックは公開行動からのクリーンルーム再構成です。

## 実リポジトリで必ず行う検証

```bash
python -m py_compile agents/kashiwashira_spidops_v1/main.py \
                     agents/kashiwashira_spidops_v1/policy_base.py
python validate_agent.py agents/kashiwashira_spidops_v1
pytest -q agents/kashiwashira_spidops_v1/tests
```

その後、席順を交互にした同一seed比較を最低200戦行います。

1. 再現版 vs 現在のv8
2. 再現版 vs v3
3. 再現版 vs genericだけでなく、上位ログに多い主要デッキ別ポリシー

## 合格基準

- クラッシュ、違法選択、観測変換fallback：0
- 攻撃／試合：5.0以上
- Spidops進化／試合：2.2以上
- トラッシュ草加速／試合：2.0以上
- 第2自ターン最終盤面：平均4.8体以上
- Mewtwo追加エネルギー：160で足りる時は0枚
- Mewtwo追加エネルギー：220/280でKOへ変わる時だけ必要最小枚数
- Giovanni使用後の同ターン攻撃率：80%以上
- 攻撃可能なのにEND：ほぼ0

勝率だけでなく、これらの中間指標が代表ログへ近いかを確認してください。異なる対戦相手分布では64%勝率や1255.2 Ratingの直接再現は保証できません。
