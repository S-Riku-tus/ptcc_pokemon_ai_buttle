# Validation — alakazam741_v12_top_sync v12.0.0

## 実行済み

- `main.py`、`policy_base.py`、テスト、比較ハーネスのPython構文検査。
- API互換Golden-stateテスト42件。要求された20状態をすべて含む。
- v11既存25件とv12 42件の合計67件を同一実行で通過。
- 現行エンジンカード表に対する全ID検査。
- 60枚、基本Energy以外4枚以下、ACE SPEC合計1枚以下の起動時検査。
- v11との`policy_base.py`バイト一致検査。
- `main.py`の重複関数定義検査、未使用の旧分岐・旧ターン状態の除去。
- 公式ローカルCGエンジンでv12対v11を200試合、100試合ずつ席順交換。
- 対戦中のcrash 0、illegal selection 0、攻撃可能END 0、0 damage Powerful Hand 0。
- 提出ZIPのroot fileが`main.py`、`policy_base.py`、`deck.csv`の3件だけであることを確認。
- 提出ZIP解凍後に現行公式APIでimport・60枚deck requestを再確認。
- 完全版ZIP解凍後に42 Golden-stateとPython構文検査を再実行。

## Golden-state

次を個別に固定した。

1. 60枚かつEnriching Energy 1枚だけがACE SPEC。
2. v11の旧ACE依存がコード・デッキにない。
3. 最初の自ターンにAbra 2体なら3体目を高評価。
4. 初回同ターンattackを作るCandyがKadabraより上。
5. Active KOと`backup_eta <= 1`成立後はsearchよりattack。
6. EnrichingをDunsparceへ付けられる。
7. Enriching付きDudunsparce循環を高評価。
8. `DRAW_ONLY` Fezへ部分給エネしない。
9. effect lockかつ短い完成ETAならFezを育成。
10. 完成資源がないFezへ1枚だけ付けない。
11. Bossは同ターンKO・勝利・保護役除去だけ高評価。
12. Boss解決後はattackを強制。
13. Hammerで攻撃効果保護Energyを剥がす。
14. 1 Energy不足でも固有効果・実attack停止価値を認識。
15. 非ミラーXerosicはattack/KO接続時だけ許可。
16. `LOCKED`中のXerosic単独使用を拒否。
17. Shayminは即時bench KO threatだけで展開。
18. Dudunsparce能力後のDunsparce再展開を優先。
19. 攻撃可能ENDを拒否。
20. 0 damage Powerful Handを拒否。

さらにBoss解決中のSupporter flag、Hammer解決中の対象価値、最後の1体Dudunsparce、Nighttime Mine、
Night Stretcher仮想改善、未知ID、5枚投入、ACE SPEC重複、重複関数定義を回帰テストした。

## 200試合ローカル比較

`benchmark_v12.py --games 200 --seed 741`の結果。searchはPoffin、Poké Pad、Hilda、Dawn、
Dudunsparce ability、Fezandipiti abilityの選択回数で数えた。

| 指標 | 目標 | v12 | v11 | 判定 |
|---|---:|---:|---:|:---:|
| 最初の自ターン終了時Abra | 2.30以上 | 2.50 | 2.48 | 達成 |
| 最初の自ターン終了時盤面 | 3.50以上 | 3.49 | 3.82 | 未達 |
| 2回目自ターン終了時Alakazam | 0.45以上 | 0.41 | 0.41 | 未達 |
| 平均初attack自ターン | 参考 | 2.43 | 2.23 | — |
| 全自ターンattack率 | 70%以上 | 60.42% | 74.31% | 未達 |
| Alakazam attack/試合 | 4.0以上 | 2.55 | 4.20 | 未達 |
| search/attack | 1.20以下 | 1.86 | 2.16 | 未達（v11改善） |
| Alakazam attack時平均手札 | 11–14 | 10.52 | 16.81 | 未達（v11過剰を解消） |
| 平均過剰damage | 100以下 | 64.93 | 183.34 | 達成 |
| 山札切れloss | 5%以下 | 1.5% | 8.0% | 達成 |
| 盤面全滅loss | 3%以下 | 27.0% | 3.5% | 未達 |
| Boss同ターンattack率 | 75%以上 | 100% (153 uses) | N/A | 達成 |
| Hammer後の相手次turn attack率 | 40%以下 | 47.40% (154 uses) | 33.33% (6 uses) | 未達 |
| Fez給エネ試合のattack到達率 | 70%以上 | N/A (0 funded) | 2.25% | 未観測 |
| Shayminが実際に防いだbench KO | 参考 | 0 (1 deployed game) | 0 | 未観測 |

このA/Bではv12が27.5%、v11が72.5%だった。v12はsearch量、手札、過剰damage、山札切れ、Boss接続を
意図どおり改善した一方、Alakazam継続attackと盤面全滅は目標に届かなかった。結果を成功扱いに
書き換えず、`validation_metrics.json`へ生値を保存した。

## seed制約

席順は厳密に100試合ずつ交換し、Python側seedは741に固定した。ただし公式native `battle_start` APIに
shuffle RNGのseed setterがないため、同一shuffle seedのペア再現はできなかった。
`validation_metrics.json`にも`same_shuffle_seed: false`として記録している。

## 実施していないこと

- Kaggle Validation Episodeへの実提出。
- 最新1位版ソースとのコード比較（公開replayのみ使用）。
- effect-lock専用対戦セットでのFez攻撃到達率70%の統計確認。
- 実bench-damage対面セットでのShaymin防止数の統計確認。
