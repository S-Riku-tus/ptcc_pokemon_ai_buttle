# MLフーディン：691.7結果後の進め方

## 現在版の位置づけ

今回の45試合では、MLによって序盤のケーシィ展開は改善した一方、山札管理と中盤の攻撃継続は改善しなかった。
特に、ノココッチ・キチキギスexの特性92回がすべてML判断であり、山札4〜6枚・手札25〜30枚でも追加ドローを選ぶ事例があった。

このため、再学習前のruntimeは「MLに広く任せる版」ではなく、「fallbackが安全な局面と判定した後、低リスク候補だけをMLで順位付けする版」とする。

## Phase 1：このguarded runtimeを基準版として保存

- デッキ60枚は変更しない。
- `ranker_model.json`も変更しない。
- 特性、END、トレーナー、エネルギー、妨害、退避はfallbackへ戻す。
- MLはケーシィ・ノコッチの展開、進化、fallbackも攻撃を選んだ場合の攻撃候補だけに限定する。
- 旧ML版は比較用に別名で保存し、上書きしない。

目的は「MLの効果を消す」ことではなく、悪化原因を切り離し、再学習前の安全なchampionを作ること。

## Phase 2：今回のログを学習データへ追加

今回の45試合を単純に全て正解教師として追加しない。

### 教師へ入れやすい行動

- 勝利試合の序盤ケーシィ展開
- 直後にフーディン完成・攻撃・KOへつながった展開と進化
- fallbackとMLが一致し、その後も攻撃継続した判断
- 上位教師と同じ方向の判断

### 除外または減量すべき行動

- 山札10枚以下の任意ドロー
- 山札切れ敗戦の終盤特性
- キチキギスexを出したが攻撃経路にならなかった判断
- 攻撃可能なのにENDした判断
- 直後に攻撃停止・盤面全滅へつながった判断
- 敗戦試合を一律に正解扱いすること

各decisionへ最低限、`game_result`、`turns_to_deckout`、`turns_to_win`、`attack_next_turn`、`ko_within_one_turn`、`board_wipe_within_one_turn`、`agent_version`を評価用metadataとして持たせる。ただし勝敗などの未来情報はpolicy featureへ入れない。

## Phase 3：学習側で追加する特徴量

優先度P0：

- `turns_to_win`
- `turns_to_deckout`
- `optional_spend_ok`
- `backup_eta`
- `attack_reserved`
- `current_ko_reserved`
- `effect_prevented`
- `fez_mode` (`NONE` / `DRAW_ONLY` / `ALTERNATE_ATTACKER`)
- `fez_completion_eta`
- `bench_threat_kind` (`direct_damage` / `damage_counter` / `ability_effect`)
- `candidate_improves_route`

カードIDの相関だけに、山札切れ・ダメージとダメカン・技術ポケモンの役割を学ばせない。

## Phase 4：action type別に段階的に再解禁

一度に全actionをMLへ戻さない。

1. `bench`（ケーシィ・ノコッチ）
2. `evolve`
3. `attack`
4. 安全条件付き`trainer`
5. 安全条件付き`energy`
6. 最後に`ability`

`END`、Boss、Xerosic、Hammer、Retreat、nested selectionは当面RULE_ONLYを維持する。

特性を再解禁する条件：

- 山札10枚以下の危険なoverrideが0
- 山札切れ率5%以下
- action type holdoutで十分なprecision
- v12 fallbackより攻撃継続が改善
- 200戦以上のChampion–Challengerで昇格条件を満たす

## Phase 5：Champion–Challenger評価

比較対象：

- Champion：今回のguarded runtime
- Challenger：新ログを追加して再学習したモデル
- Baseline：純粋なv12 fallback

同一seed・席順交換で最低200戦。勝率だけでなく次を比較する。

- 全自ターン攻撃率
- フーディン攻撃／試合
- 初攻撃後の非攻撃ターン
- 山札切れ
- 盤面全滅
- 山札10枚以下の任意ドロー
- キチキギスexの展開→給エネ→攻撃到達率
- ML action type別override数と、その後1ターン以内の攻撃・KO率
- crash、illegal action、timeout

昇格の最低条件例：

- 200戦以上
- guarded runtimeに対して勝率53%以上
- crash 0、illegal action 0
- 山札切れ5%以下
- 攻撃率70%以上
- 敗戦時の初攻撃後停止0.8以下

## Phase 6：Kaggle提出

1. 旧ML、guarded、再学習challengerを別agent名で保存する。
2. ローカル比較で勝ったものだけをKaggleへ提出する。
3. 最初の30〜50試合は安全性を確認する。
4. 100試合以上で対面別・action type別に再分析する。
5. Rating単独ではなく、上記行動指標と合わせて採否を決める。

## シェイミについて

ドラパルトexのベンチへの「ダメカンを置く」効果は、シェイミの攻撃ダメージ防止では防げない。したがって、ドラパルト対策としてシェイミ展開を強制しない。

シェイミは、相手Activeが直接ベンチダメージ技を次ターン使用可能で、こちらの非ルールポケモンが実際にKO圏であり、展開しても攻撃・ケーシィ枠を失わない場合だけfallbackが出す。

## 最終方針

次の一手は、現在モデルへさらに例外を足すことではない。

1. runtimeを安全に縮小する。
2. 今回ログを成果付き・失敗付きのデータとして追加する。
3. 戦略特徴量を増やして再学習する。
4. action typeを段階的に再解禁する。
5. guarded runtimeをchampionとして比較する。

この順序なら、MLによる序盤展開の改善を残しながら、山札管理や技術ポケモンの役割をfallbackで守れる。
