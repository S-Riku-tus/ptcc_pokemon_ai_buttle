# v30 変更点

## データ

- 教師コーパスを361試合から2,361試合へ拡張
- Majkel1337とYushin Itoの各1,000試合を統合
- 同一デッキハッシュのみを採用
- 教師、順位、コホート、episode順を保持するindex builderを追加

## 推論

- 全context対応の圧縮教師メモリを追加
- カード個体差を吸収する意味的行動復元を追加
- 凍結v29を明示的な親方策として同梱
- v29候補スコアを入力にする残差LambdaRankへ変更
- v30、v29、従来ranker、決定論fallbackの段階的fallbackを実装
- v29のリーサル・KO・Bossルート・盤面枚数安全条件を継承

## 特徴

- 422特徴から648特徴へ拡張
- 公開行動履歴とターン内行動系列
- 各盤面枠のエネルギー、どうぐ、進化元
- search時の公開deck候補
- 合法手集合の構成
- v29の選択、スコア、gap、rank

## 学習・評価

- コホート内時系列70/10/20 splitへ変更
- 意味的重複候補318,421件を統合
- 121,708 MAIN盤面、1,081,441候補で学習
- 既知ログのランタイム再現評価を追加
- 未見top-1/top-2/top-3と行動種別別評価を追加

## 開発補助

- `build_alakazam_v30_teacher_corpus.py`
- `audit_alakazam_v30_teacher_policy.py`
- `build_alakazam_v30_teacher_memory.py`
- `train_alakazam_v30_teacher.py`
- `evaluate_alakazam_v30_teacher_index.py`
- 教師メモリ、search選択、履歴特徴、artifact固定を検査するv30テスト
