# v2 検証結果

## 実施済み

- `main.py`と`policy_base.py`のPython構文検査：成功
- API互換スタブによる単体・統合テスト：26件成功
- 60枚と代表提出`54603674`のデッキ一致：確認
- 提出版`54655021`のdeck snapshotとv1ソースのSHA-256一致：確認
- 次の失敗原因をテストで固定
  - Poké Pad候補がすべて負になる問題
  - Poké Padがサポーター使用後に禁止される問題
  - Factoryが手札7枚以上で止まる問題
  - 非KO Giovanni
  - MimikyuからSpidopsへのピボット
  - 非KO時はエンジン、KO時は攻撃というMAINフェーズ順序
  - CrustleへMewtwo exで攻撃する問題
  - Alakazam対面のArticuno優先
  - 高速ex対面の2本目進化ライン
- コンパイル済みキャッシュを提出ZIPから除外
- 提出ZIPのルート直下が`main.py`、`policy_base.py`、`deck.csv`のみであることを検査

## 通し監査で確認した不変条件

1. 勝利KO > 確定KO > KOを作る行動 > 安全なエンジン行動 > 非KO攻撃 > その他
2. Poké Padはサポーター使用状況に依存しない
3. GiovanniはKOできない場合に使用しない
4. Factoryは山札2枚以下だけ停止する
5. 進化や検索は確定KO Tierを超えない
6. Mewtwo exはCrustleへ0ダメージとして扱う
7. 退避はベンチの攻撃が実際に改善する場合だけ行う
8. すべての選択は`policy_base.py`の合法フォールバックを保持する

## 未実施

この環境には実プロジェクトの`vendor/cg`、`validate_agent.py`、対戦ハーネスがないため、実エンジン対戦とRating検証は未実施です。

## 実リポジトリでの必須確認

```bash
python -m py_compile runs/kashiwashira_spidops_reconstruction_v2/main.py \
                     runs/kashiwashira_spidops_reconstruction_v2/policy_base.py
python validate_agent.py runs/kashiwashira_spidops_reconstruction_v2
pytest -q runs/kashiwashira_spidops_reconstruction_v2/tests
```

その後、v1とv2を同じ対戦相手・seed・席順交換で最低200戦比較してください。

## v2の目標値

- Poké Pad取得率：ほぼ100%
- Factory提示時選択率：85%以上
- Spidops進化：2.6回／試合以上
- Spidops加速：2.5回／試合以上
- T4・T5攻撃率：40%以上
- Giovanni同ターンKO率：75%以上
- 攻撃した自ターン：4.3回／試合以上
- クラッシュ・違法選択・fallback：0
