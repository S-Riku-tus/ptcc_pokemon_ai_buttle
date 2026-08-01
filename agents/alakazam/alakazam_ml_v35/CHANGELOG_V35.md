# CHANGELOG v35

対象: `agents/alakazam/alakazam_ml_v35`
基盤: `alakazam_ml_v34`
作成: 2026-08-02

## 変更（2件、いずれもランタイム）

### 1. `lethal_guard` を強制から禁止へ

`ml_runtime.HybridRanker.choose` は、v29 ベースラインの手が推定致死の Powerful Hand であるとき、ランカーを呼ぶ前にベースラインへ差し替えていました。v35 はフラグだけ立てて先に進み、ランキング後のガードで判定します。

発火条件: ランカーの選択が `end`（`lethal_declined_by_end`）または非致死の `attack`（`lethal_declined_by_weak_attack`）のとき。

根拠: `attack_lethal_estimate` は「相手アクティブを倒せる」であって「勝利確定」ではないため、test 9,977判断のうち1,122件（1試合あたり約6回）で発火していました。ランカーの一致率79.7%に対し強制後は56.4%、純損失261判断です。待つことのリスク（Powerful Hand の火力＝20×手札枚数が下がる）は `breaks_current_ko` が既に禁止しているため、この強制は冗長でした。

### 2. `preserve_fallback_boss_route` をターン終了時のみに

v29 ベースラインが Boss's Orders を選び、ランカーが選ばなかったとき、常にベースラインへ差し替えていました。v35 はランカーの選択が `end` のときだけ発火します。

根拠: 326件発火し、強制された Boss の一致率は13.2%。ランカーなら64.7%で、純損失168判断です。ランカーが非終了の手を選んだ場合、Boss はターン内で先送りされるだけでルートは失われません。

### 実装

狭めたガードは `ml_runtime._v35_safety_reason` に独立して置きました。`v29_runtime._candidate_safety_reason` は**変更していません**。v29 ベースラインの選択そのものがランカーの入力特徴（`v29_selected`、`v29_ranker_score`、`v29_ranker_rank`）なので、共有関数を変えるとコーパスがモデルの下からずれます。

`ALAKAZAM_ML_V35_SHELL=v34` で旧シェルに戻せます（同一モデルでの A/B 用）。環境変数の読み取りは `ALAKAZAM_ML_V35_*` を先頭にした版チェーンに整理し、v31〜v34 の名前も引き続き有効です。

## 変更していないもの

- `ranker_model.json`（sha256 `0a19b5cd…`、2,050本、seed 1091、657特徴）
- `v29_ranker_model.json`、`legacy_ranker_model.json`、`target_ranker_model.json`、`teacher_memory.bin`、`fallback_policy.py`、`deck.csv`（全ハッシュ検証済み）
- コーパス、ホールドアウト境界、graded ラベル、recency 重み
- `breaks_current_ko`、`end_with_ready_attack`、`dudunsparce_body_floor`、`unmodelled_other`、v29 フォールバック、盤面メモリ

## 測定結果

test 200試合・9,977判断、モデルは学習ブロックのみで再学習した held-out 版。

| 指標 | v34 | v35 | 差 |
|---|---:|---:|---:|
| ランカーの Top-1 | 83.01% | 83.01% | 0 |
| **実際に打つ手の Top-1** | **77.85%** | **82.14%** | **+4.29** |
| 実際に打つ手の turn-set | 94.46% | 95.22% | +0.76 |
| ランカー破棄率 | 16.28% | 1.77% | -14.51 |

validation: 76.73% → 81.92%。

実エージェントでの end-to-end A/B（60試合・4,113判断、シェルのみ切り替え）: 87.79% → 92.71%、フォールバック率 10.75% → 1.43%。

推論: 98.9 ms → 106.2 ms（+7.4%）、モデル選択率 81.7% → 95.8%。

## テスト

`test_v35_runtime_logic.py` を追加（14件）。

- 致死盤面で非終了の手が通ること、`end` と非致死攻撃は依然拒否されること
- 致死盤面でも `breaks_current_ko` が効くこと（緩和の根拠そのもの）
- Boss ルートが `end` 以外を阻まなくなったこと、`end` には効くこと
- 変更していない4ガードの挙動
- 狭めた関数が共有版と、意図した2箇所以外で一致すること（総当たり）
- 既定シェルが v35 であること、`ALAKAZAM_ML_V35_SHELL=v34` で戻せること
- 配備モデルが v34 とバイト同一であること

`test_v15_runtime_logic.py` の環境変数テストはソース文字列の grep から挙動検査に置き換えました（v35 は版チェーンをループで読むため、リテラル名がソースに現れません）。意図は同じで、v31〜v35 のすべての名前が有効であることを実際に確認します。
