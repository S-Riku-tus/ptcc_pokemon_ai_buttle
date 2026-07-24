# v19 → v20 独立設計分析（第2 AI）

作成日: 2026-07-24

対象: `agents/alakazam/alakazam_ml_v19/`

目的: デッキ60枚と学習済みランカーを変更せず、Boss、手札最適化、ターゲット別KOルートを決定論ロジックで改善する。

## 0. 結論

優先順位は次の通り。

1. **P0: 終局 Boss ゲートをスコア計算の最上段に追加する。**
   現行にも「Boss後の手札減少」「効果防止」「プライズ値」を考慮する良い部品はあるが、
   「残り2/3サイドをBenchの1体で取り切れるなら、Bossを必ず選ぶ」という選択全体の
   不変条件はない。既存部品を束ねる小さな変更で実現できる。
2. **P1: 先にターゲットを決め、そのターゲットの必要手札に達したら純ドローを止める。**
   一律の手札上限は採用しない。v19実ログでは Run Away Draw 90回中35回
   （38.9%）が「相手Activeは既に現在手札でKO可能」な状態だった。必要手札到達後だけ
   止めるターゲット依存ゲートなら、既知の「広いデッキアウトガードはcabt勝率を下げた」
   後退と切り分けられる。
3. **P2: `_active_route_plan()` を `target` 引数付きの汎用KOルートへ一般化する。**
   ターゲット優先順位とルート可否を分離し、優先リストの先頭から
   「Active直撃 / Hammer解除 / 非Supporterドロー / Boss / attack」を探索する。
   初版は現在の確定手札増加だけを扱い、未知ドローの中身は仮定しない。

今回の3要件に **ML再学習は不要**。`main.py:17-20` の既定閾値は0.37だが、
`ml_runtime.py:19-35` で Boss、attack、Trainer、Energy、Ability、Retreat は
決定論側に固定され、ライブ上書き対象は `{"bench", "evolve"}` だけである。
さらに fallback が attack 等を選んだ判断全体は `ml_runtime.py:139-155` でML対象外になる。

## 1. 調査方法と定量監査の注意

Claudeの一次分析は、この節から第7節までを固める間は読んでいない。
独立分析の入力はv19コード、v19設計履歴、52戦の自分ログ、上位ログ、既存の
サイド取得レポートだけである。Claudeとの比較は最後の第8節でのみ行う。

新規のログ監査は、既存の再生方法
（ある観測に対する選択は次のreplay stepの `action` に入る。
`scripts/analyze_alakazam_policy_imitation.py:109-111`）と同じ方法で行った。

- 自分: v19 submission 54938555、52戦、30勝22敗。
- 比較用上位: rank 2/3/5/8、同一主流デッキ、261戦。
- Powerful Handの過剰火力は、攻撃時の相手の**残HP**と `20 × handCount` の差で計算。
- 勝ち確Boss機会は、BossがMAINで合法、残りサイドが2または3、Active 743が攻撃可能、
  Bossを手札から使った後の火力 `20 × (handCount - 1)` がBench対象の残HP以上、
  `prize_count(target) >= remaining_prizes`、可視の自己/エネルギー/全体効果防止がない状態。
- Bench対象には「前の攻撃による一時無敵」は適用されない。v19自身も一時無敵を
  現在の相手Activeと同一個体の場合だけ認める
  (`fallback_policy.py:1132-1146`)。
- 「現在Activeが既に致死の時のドロー」は無駄の**代理指標**である。進化は後続育成、
  Dawn/Hildaは盤面構築の価値も持つため、すべてを誤手とは数えない。
- 上位との使用回数比較には対面・試合長・到達状態の交絡がある。

## 2. アーキテクチャ確認

### 2.1 fallbackが先、MLは狭い上書き

`main.py:23-27` はまず `_fallback_agent()` の選択を取得し、その後に
`HybridRanker.choose()` を呼ぶ。`ml_runtime.py:19-28` の `RULE_ONLY_ACTIONS` には
`ability/end/trainer/energy/boss/retreat/xerosic/hammer/other` が入り、
`ml_runtime.py:35` のライブ許可は `bench/evolve` だけである。

fallback本体は `fallback_policy.py:1986-2000` で全オプションを `_score()` し、
最大スコアを選ぶ。中心dispatchは `fallback_policy.py:2222-2308`。

したがって今回の中心は次の2点で完結する。

- fallbackに終局ゲート、ターゲット計画、必要手札ゲートを追加。
- `evolve` が選択された場合だけはMLが進化先を変え得るので、fallbackがattack/Bossを
  選ぶべき局面では先にそれらを最高点にする。

### 2.2 カードDBとプライズ値

`policy_base.py:33-51` はカード/攻撃/エネルギー表を構築し、
`policy_base.py:178-180` の `prize_count()` は Mega ex=3 / ex=2 / その他=1。
ただし実戦の `fallback_policy.py` は `BasePolicy` のサブクラスではなく、
同じDBとヘルパーを自己完結で複製している
(`fallback_policy.py:228-236`, `:469-471`)。

v20で実際に変更すべきなのは `fallback_policy.py` 側である。
`policy_base.py` だけを直しても今回のfallback判断は変わらない。

## 3. 要件1: Boss's Orders の絶対条件

### 3.1 v19の現行挙動

現行Boss処理はかなり洗練されている。

1. **Boss使用後の実火力**
   - `_boss_damage_after_spend()` (`fallback_policy.py:1460-1478`) は、
     MAINでBossを使う前なら手札を1枚減らして 743 の火力を計算する。
   - Bossの対象選択中はカードが既に手札を離れているため二重に減らさない
     (`:1463-1477`)。
   - 743以外は `_active_best_dmg()` に委譲する。
2. **効果防止**
   - 自己防止、Mist等のエネルギー、全体保護を
     `_non_energy_effect_prevented()`、`_static_effect_prevented()`、
     `_effect_prevented()` で判定する (`:1148-1178`)。
   - Powerful Handは効果防止対象なら0ダメージ
     (`:1246-1254`)。
3. **ターゲット価値**
   - `_boss_role_bonus()` (`:1753-1779`) は既存の
     `BOSS_KEY_ROLE_BONUS`、進化段階、ex、エネルギー、保護/ロック能力を評価する。
   - `_target_value()` (`:1781-1796`) はプライズ、エネルギー、Tool、役割、進化を加点。
   - `_boss_target_score()` (`:1862-1948`) は同ターンKO、勝利、
     ActiveのKOとの比較、保護ロック脱出、2プライズ以上等を判定する。
     `winning = target_prizes >= len(self.me.prize)` は `:1885-1901` に既にある。
4. **Bossカードと対象選択**
   - Bossカード自体は `_score_play_trainer()` の
     `fallback_policy.py:2701-2711` で評価する。
   - Boss解決後の相手Active選択は `_score_active_choice()` の
     `:3252-3257` から同じ `_boss_target_score()` を使う。
5. **現在Activeの終局攻撃**
   - `_terminal_win_attack_offered()` (`:2563-2581`) はプライズ勝利または
     相手最終ボディKOを検出する。
   - それが提示されると、`_score()` はATTACK以外を一律 `-1`
     (`:2243-2246`) にする。
6. **未接続ヘルパー**
   - `_winning_gust_ready()` (`:2606-2614`) は似た条件を持つが呼ばれていない。
     しかもBoss消費前の `_active_best_dmg()` を使うため、手札1枚減を反映しない。
     v20でこのまま再利用してはいけない。

### 3.2 gap

現行には「勝ち確Bossを選択全体の最上位にするゲート」がない。
Bossは通常のスコア競争に参加するだけである。

- `_boss_target_score()` の勝利加点は90,000だが、BossカードのMAINスコアは
  `18500 + min(12000, best // 8)` なので最大30,500
  (`fallback_policy.py:2704-2711`)。
- Rare Candyの即KOは33,000 (`:2650-2654`)、
  Active確定ルート行動は非終局47,000、終局88,000 (`:1732-1739`)。
  後者が同じく勝利に直結するなら勝敗上は問題ないが、
  ユーザー指定の「Bossを絶対条件にする」は満たさない。
- 残りサイドを2/3に限定した明示ゲートもない。現行の
  `target_prizes >= len(self.me.prize)` はより一般的だが、スコア内部の条件でしかない。

ログでは以下だった。

| 指標 | v19 52戦 | 上位4 261戦 |
|---|---:|---:|
| Boss使用 | 27 | 188 |
| Bossと同ターン攻撃 | 27/27 (100%) | 149/188 (79.3%) |
| 厳密な残2/3サイド勝ち確Boss機会 | 1ターン | 24ターン |
| 同ターンBoss実行 | 1/1 | 7/24 (29.2%) |
| 1プライズActiveを攻撃し、KO可能な2+プライズBenchを見送った状態 | 0 | 10 |

v19自身の52戦には「勝ち確Bossを見送った」実例はなかった。したがってP0は
52戦の再発バグ修正というより、低頻度だが勝敗に直結する仕様の形式化である。
上位も17/24を見送っており、これは上位模倣ではなくユーザーが明示した戦略変更になる。

代表例:

- 上位 episode `87660233` turn 13: 残り2、手札13、Boss後240、
  Bench Fezandipiti ex(140)残HP210で即勝ち可能だったが、
  Dudunsparce進化 → Run Away Draw → Lana's Aidを選び、その試合は敗戦。
- 上位 episode `87635770` turn 18: 残り2、相手Activeは単プライズ743残HP140、
  Bench Fezandipiti ex(140)残HP10。Bossで即勝ち可能だったがActiveを攻撃した。

### 3.3 実装提案

既存名と区別するため、以下は **v20で新設する関数名案** である。

```python
def _terminal_boss_targets(self):
    if self.context != SelectContext.MAIN:
        return []
    if self.state.supporterPlayed:
        return []
    remaining = len(self.me.prize)
    if remaining not in (2, 3):
        return []
    if not self._main_has_card_action(C.BOSS_ORDERS, (OptionType.PLAY,)):
        return []

    targets = []
    for target in self.opponent.bench:
        if target is None:
            continue
        # このhelperは防止なら0、MAINではBossの手札1枚分も引く。
        damage = self._boss_damage_after_spend(target)
        if (prize_count(target) >= remaining
                and damage >= max(1, target.hp)):
            targets.append(target)
    return targets
```

`_score()` のMAIN dispatch直前、少なくとも現在の
`_terminal_win_attack_offered()` ゲート (`fallback_policy.py:2243-2246`) と同じ高さに置く。

```python
terminal_boss = self._terminal_boss_targets()
if terminal_boss:
    if t == OptionType.PLAY:
        card = get_card(self.obs, AreaType.HAND, o.index, self.my_index)
        if card is not None and card.id == C.BOSS_ORDERS:
            return 120_000
    return -1
```

Boss解決後の `TO_ACTIVE` では、候補が
`prize_count >= remaining` かつ `_boss_damage_after_spend >= hp` なら
通常の役割点より上の `120_000 + prize_count * 100` にする。
同点は `_target_value()`、残HP、安定したserial順で決める。

#### Activeも既に勝ちの場合

仕様を字義通り実装するならBossを選ぶ。ただし、現在Activeへの攻撃も即勝利なら
Bossは余分なゲーム操作で、勝率上の利得はない。技術的な推奨は次の優先順位である。

1. 現在Activeへの提示済み攻撃で即勝利。
2. それがなければ勝ち確Boss。
3. その他。

ユーザー原文を厳密に優先する統合実装では1と2を逆にできる。この差は必ずgolden testで固定し、
暗黙のスコア順に任せない。

#### 必須テスト

- 残2、Bench ex、Boss後火力でKO可能 → Boss。
- 残3、Bench Mega ex、Boss後火力でKO可能 → Boss。
- 残3、Bench ex（2プライズ）だけ → 絶対ゲート不発。
- `20 × handCount` では届くが `20 × (handCount - 1)` では届かない → Bossしない。
- Mist/自己防止/全体保護 → Bossしない。
- Supporter使用済み、Boss非提示 → Bossしない。
- Boss解決後は手札を二重に1枚減らさない。
- Activeも即勝利の時の優先順を仕様通り固定。

## 4. 要件2: 手札枚数のより良い最適化

### 4.1 v19の現行挙動

v19は「手札を増やす」制御は強いが、「選んだ相手に必要なところで止める」制御が弱い。

#### 既にある良い制御

- Powerful Handの実火力は `20 × handCount`
  (`fallback_policy.py:1246-1254`)。
- 致死攻撃が提示済みの時、手札を減らす行動で致死を失わないようにする
  (`:2247-2276`)。
- 攻撃可能なポケモンへの過剰エネルギー貼りを止め、エネルギーを手札に残す
  (`:3029-3041`)。
- Enriching Energyだけは相手Activeの残HPから
  `required = ceil(hp / 20)` を計算し (`:2907-2923`)、
  現在KOと後続が確保済みなら追加4ドローを止める。
- Active限定の確定KOルートは、手札純増を
  Dudunsparce +3、Enriching +3、Alakazam進化 +2、
  Kadabra進化 +1、Rare Candy +1、Dawn +2、Hilda +1 として探索する
  (`:1514-1602`, `:1639-1723`)。
- `_deck_floor()`、`_deck_spend_ok()` は低山札で任意ドローを止める
  (`:541-562`)。

#### 攻撃的なドロー

- Bench DudunsparceのRun Away Drawは通常15,000点
  (`fallback_policy.py:2373-2387`)。
- 止まるのは主に `_deck_preserve()`、手札12以上かつ山札14以下、
  またはデッキフロア。
- Hilda/DawnはActiveが到達可能な時に14,000/13,800
  (`:2664-2689`)。
- `_ko_active_reachable()` と `_achievable_hand()` は相手Activeだけを見る
  (`:2455-2463`, `:2616-2623`)。

#### 手札差分の不整合

- `_active_route_atoms()` はDawnを純増+2と正しく扱う
  (`fallback_policy.py:1581-1591`)。
- 一方 `_hand_delta()` はHildaとDawnを両方+1にする
  (`:1974-1979`)。
- `_achievable_hand()` もHilda/Dawnを一律+1とし、Dudunsparceは1体分だけで、
  進化・Rare Candy・Enrichingを含まない (`:2455-2463`)。

同じ概念が3か所で別の近似になっている。v20では一つの
`projected_hand_delta` に統合すべきである。

### 4.2 定量gap

| 指標 | v19 52戦 | 上位4 261戦 |
|---|---:|---:|
| MAINでRun Away Draw選択 | 90 (1.73/戦) | 776 (2.97/戦) |
| その時点で相手Activeが既にPowerful Hand致死 | 35/90 (38.9%) | 124/776 (16.0%) |
| Powerful Hand KO攻撃 | 161 | 875 |
| KO時平均過剰火力 | +140.1 | +115.4 |
| 必要手札より3枚以上多いKO | 113/161 (70.2%) | 600/875 (68.6%) |
| 必要手札より5枚以上多いKO | 96/161 (59.6%) | 446/875 (51.0%) |
| クセロシキ被弾 | 22 (0.42/戦) | 172 (0.66/戦) |
| 被弾時に3枚まで落とされた推定枚数 | 平均10.6 | 平均8.0 |
| デッキアウト負け | 3/52 (5.8%全試合、13.6%敗戦) | 8/261 (3.1%) |

重要な解釈:

- 「上位はRun Away Drawの総発動が少ない」は、生の実行回数/試合では再現しなかった。
  上位は試合当たりではむしろ多い。
- 再現した差は **上位は現在Activeの致死到達後にRun Away Drawを選ぶ割合が低い**
  ことである。v20で模倣すべきなのは固定回数や固定手札上限ではなく、この条件付き停止。
- v19には、現在Active致死後の確定手札増加が合計105回あった
  （Dawn 36、Run Away Draw 35、Kadabra進化17、Rare Candy 13、
  Enriching 2、Alakazam進化2）。Run Away以外は後続育成を兼ねるため、
  選択KOルート外かどうかを追加条件にしなければならない。
- KO時の大きな手札は、それ自体では損失ではない。既に手札にあるカードを捨てて
  最小値へ合わせる必要はない。抑えるべきなのは、目標達成後のデッキ消費と
  不要な盤面露出である。
- v19は52戦で22回クセロシキを受け、すべて手札4枚以上だった。被弾すると
  Powerful Handは原則60まで落ちる。開始手札8でも20でも結果は3枚なので、
  「クセロシキ対策として余分に何枚かバッファする」は成立しない。

### 4.3 必要手札の式

743のPowerful Handで対象を倒す、攻撃直前の必要手札は:

```text
attack_hand_required(target) = ceil(max(1, target.hp) / 20)
```

現在のMAIN手札から直ちに行う場合:

```text
Activeを攻撃:
  pre_action_hand_required = attack_hand_required

BossでBenchを呼んで攻撃:
  pre_action_hand_required = attack_hand_required + 1

Boss + Enhanced Hammer:
  pre_action_hand_required = attack_hand_required + 2
```

より一般には:

```text
projected_hand_at_attack
  = current_hand
  + sum(guaranteed_net_draw)
  - mandatory_cards_spent

surplus = projected_hand_at_attack - attack_hand_required(target)
```

Powerful Handを防ぐ効果が残る場合、必要手札は有限値ではなく「到達不能」。
Hammer等の解除をルートに含めてから再計算する。
245のPsychic等はこの式を使わず、既存 `_route_attack_damage()` に委譲する。

### 4.4 実装提案

#### A. 手札射影を一元化

新設案:

```python
def _guaranteed_hand_delta(self, kind, option=None):
    return {
        "dudun": 3,
        "enriching": 3,        # 1枚貼る、4枚引く
        "evolve_alakazam": 2,  # 1枚出す、3枚引く
        "evolve_kadabra": 1,   # 1枚出す、2枚引く
        "rare_candy": 1,       # Candy+Alakazamを出す、3枚引く
        "dawn": 2,             # 1枚使う、3枚手札へ
        "hilda": 1,            # 1枚使う、2枚手札へ
        "boss": -1,
        "hammer": -1,
        "normal_attach": -1,
    }[kind]
```

`_active_route_atoms()`、`_hand_delta()`、`_achievable_hand()` が同じ表を使う。
効果の任意ドローをNOにする可能性があるカードは、実際にYESを選ぶルートだけ純増を付ける。

#### B. ターゲット到達後の停止

新設する `_chosen_ko_plan()` が返す `target`、`actions`、`required_hand` を利用する。

```python
def _draw_needed_for_chosen_target(self, kind):
    plan = self._chosen_ko_plan()
    if plan is None or not plan["ko"]:
        return True
    if kind in plan["actions"]:
        return True
    if self._energy_starved() or self._ready_alakazam_count() == 0:
        return True
    return False
```

適用方針:

- **Bench Run Away Draw（純フィルタ）**:
  選択ターゲットが現在の手札で既に確定KO、attack/Bossが合法、解除札も不要なら `-1`。
  Active Dudunsparceの退避、唯一ボディ安全則、攻撃役への交代は既存例外を維持する。
- **Enriching**:
  既存 `_enrich_draw_needed()` をターゲット引数付きにし、選択ターゲットで判定する。
- **Dawn/Hilda**:
  選択ルートに含まれず、攻撃役・後続・エネルギーが揃っている時はKO攻撃6,000点未満へ下げる。
  一律禁止にはしない。
- **進化ドロー/Rare Candy**:
  選択ルート、最初の攻撃役、明確な1ターン後続のいずれにも不要なら攻撃より下げる。
  盤面構築価値があるため、Run Away Drawと同じハード禁止にはしない。
- **攻撃**:
  既存の「余裕があれば先に展開」思想は、選択ターゲットの最小ルート内の行動にだけ許す。

#### C. ルートのタイブレーク

現行 `_active_route_plan()` の比較キーは
`(action_count, deck_cost, supporter_cost, -damage)`
(`fallback_policy.py:1707-1712`) で、最後は過剰火力が大きい方を選ぶ。
v20は次に変える。

```text
(action_count,
 deck_cost,
 supporter_cost,
 max(0, damage - target.hp),
 irreversible_resource_cost)
```

これにより同じ手数なら「必要量に最も近い確定ルート」を選ぶ。

### 4.5 Xerosicのダウンサイドをどう扱うか

**追加の手札バッファ目標は入れない。**
Xerosicは「何枚か減らす」のでなく3枚まで落とすため、8→3も20→3も攻撃直後の火力は同じ。

入れるべき影響は間接的である。

- ミラーで必要手札に達したら、不要なドロー/展開より今の攻撃を優先する。
- ターゲット到達後にデッキを掘り切らず、次ターンの再建用ドロー源を山札に残す。
- 相手のXerosic使用が公開情報になった後は診断カウンタを持たせ、
  「被弾後何ターンで必要手札へ戻ったか」を測る。
- 相手手札の未知カードを読んでXerosic確率を作る初版は避ける。

## 5. 要件3: ターゲット優先順位 → 順番にKOルート探索

### 5.1 v19の現行挙動

ターゲット価値は現在、用途ごとに分断されている。

- Boss: `_boss_role_bonus()`、`_target_value()`、`_boss_target_score()`
  (`fallback_policy.py:1753-1948`)。
- Boss候補一覧: `_gust_ko_targets()` (`:1950-1955`)。
- Fezandipiti exの100点狙撃: `_fez_target_score()` と
  `_score_attack()` (`:1048-1065`, `:3089-3101`)。
- 通常攻撃: 相手Activeだけ (`:3076-3136`)。
- 確定KOルート: `_active_route_plan()` は相手Active固定、
  自分Active 743固定 (`:1604-1627`)。
- ルート候補はHammer、確定枚数の進化ドロー、Enriching、
  Dudunsparce、Dawn、Hilda (`:1514-1602`)。
- `_ko_active_reachable()` も相手Active固定 (`:2616-2623`)。

つまり「誰を倒すか」はBoss/Fezのローカル評価にしかなく、
各Trainer・Ability・進化は全体で共有するターゲット計画に従っていない。

### 5.2 gap

既存のサイド分析では、上位vs exで2プライズ体を取った試合は
取らない試合より平均KOが1.4少なく、1.6ターン短い。
にもかかわらず、Activeの単プライズKOルートが先に見つかると、
Benchの高プライズ体を倒すためにどこまで手札を作るかという共通計画がない。

自分52戦では「Bossで高プライズBenchを取れるのに低プライズActiveを攻撃」は0件だったが、
母数が小さい。上位4では10状態あり、残り2/3サイドでFezandipiti exやMega exを
見送る例も含まれた。高プライズを狙う価値は、模倣頻度より既存のターン/KO効率データを
優先して判断すべきである。

### 5.3 ターゲット優先リスト

新設案: `_target_priority_list()`。

入力:

- 自分/相手の残りサイド: `len(self.me.prize)`, `len(self.opponent.prize)`。
- 相手Active/Bench各個体: serial、ID、残HP、最大HP、`prize_count`、進化段階、
  エネルギー、Tool、効果防止、退却コスト。
- 既存 `_boss_role_bonus()` と `_target_value()`。
- 自分盤面: 攻撃可能な743/245、後続、手札、エネルギー、Boss/Hammerの合法性。
- 自分のプライズ負債: Fezandipiti exが場にいるか。
- 可視盤面だけからのKO回数下限:
  高いプライズ値から足して残りサイドへ届く最小個体数。

まずレース状態を作る。

```python
self_ko_lb = min_visible_kos(len(self.me.prize), opponent_board)
opp_ko_lb = min_visible_kos(len(self.opponent.prize), self_board)
race_pressure = (opp_ko_lb <= self_ko_lb)
fez_close_risk = (
    self.field[C.FEZANDIPITI_EX] > 0
    and len(self.opponent.prize) <= 2
)
```

各ターゲットの初期重み案:

| 項目 | 点 |
|---|---:|
| この1KOで勝利 | +100,000 |
| プライズ1枚ごと | +12,000 |
| race pressure中のプライズ1枚ごと | 追加+3,000 |
| 既存 `_boss_role_bonus(target)` | そのまま |
| 現在攻撃可能な脅威 | +1,500 |
| 付与エネルギー1個 | +300 |
| 現在Active（Boss不要のテンポ） | +1,000 |
| 効果防止中 | 点を下げず、ルート側で解除不能ならskip |
| 必要手札1枚ごと | -50（同価値内のtie-breakだけ） |

安定ソートキー:

```text
(
  -wins_by_this_ko,
  -strategic_score,
  -prize_count,
  required_hand,
  0 if active else 1,
  serial,
)
```

重要なのは、**ターゲット価値と到達可否を同じスコアに潰さない**こと。
優先1位が到達不能ならルート探索が2位へ進む。低HPだから最初から1位にするのではない。

ターン開始（強制1ドロー後）の最初のMAINでserial順リストを
`_V9_STATE` に保存する案が自然である。`_V9_STATE` は既にターン跨ぎの公開情報を保持し、
ゲーム開始時にリセットされる (`fallback_policy.py:169-186`, `:3549-3588`)。
新しいキャッシュも `diag_reset()` と `select is None` の両方で消す。
相手は自分ターン中に盤面を変更しないため、対象が消えた時だけ次点へ進めればよい。

### 5.4 汎用KOルート

新設案:

```python
def _ko_route_plan(self, target_ref):
    # target_ref: serial, current area, id
    # return:
    # {
    #   reachable, ko, winning, target_serial,
    #   damage, hand_at_attack, required_hand,
    #   actions (ordered tuple), action_count, deck_cost,
    #   needs_boss, needs_hammer, attacker_serial
    # }
```

一般化の順序:

1. **Active target**
   - 現行 `_active_route_plan()` をそのままtarget引数化。
2. **Bench target + Boss**
   - Bossが手札/MAINにあり、Supporter未使用。
   - 初期手札からBossの1枚を予約。
   - Bossを使うのでDawn/Hilda atomは除外。
3. **効果防止**
   - エネルギー由来で、その個体の該当EnergyをEnhanced Hammerで選べるなら
     `hammer` を必須atomにする。
   - 自己能力/全体保護で解除ルートがなければ到達不能。
4. **確定手札増加**
   - 既存 `_active_route_atoms()` の非Supporter atomを再利用。
   - Run Away Drawの唯一ボディ/Active退避安全則、
     Enrichingの1回エネルギー権、進化対象競合を維持。
5. **順序**
   - Hammer（必要時）
   - 非Supporterの確定ドロー/進化
   - Boss
   - attack
6. **次点**
   - 優先ターゲットが到達不能なら次のターゲットへ。

初版はActive 743を中心にする。既存 `_route_attack_damage()`
(`fallback_policy.py:1315-1339`) は245、Kadabra、Fezandipiti ex、Dudunsparceも扱えるので、
テストを追加した後に攻撃役一般化が可能。Fezandipiti exはBenchを直接狙えるため、
Boss不要ルートを別分岐にする。

#### スコアへの接続

`_chosen_ko_plan()` が選んだ「次に必要な行動」だけを高得点にする。

```text
terminal attack             130,000
terminal Boss（厳密仕様時） 120,000
terminal route next action   110,000
nonterminal KO route action   47,000 + prize * 2,000
route外の純ドロー            attack score未満
```

数値そのものより、終局/確定KO/準備の層を交差させないことが重要。
既存の `95000/90000/88000/47000/30000...` を定数化してから導入すると監査しやすい。

## 6. リスクと既知後退との衝突

### 6.1 一律デッキアウトガードを再導入しない

`fallback_policy.py:2373-2387` は、広いデッキアウトガードや単純手札capが
cabtで後退したため、攻撃的ドローを維持したと明記する。
一方、v19実ログにはなお3/52のデッキアウト負けがある。

この2事実から、次を分離する。

- 維持: `_deck_floor()`、`_deck_spend_ok()`、唯一ボディ安全則、Active Dudunsparce退避。
- 変更: **選択ターゲットの確定KOに必要ない純ドローだけ**を止める。
- 非採用: `handCount >= N` だけの一律cap、`deckCount <= N` だけの広い停止、
  リード中の全Trainer停止。

これは「山札保護」ではなく「最小確定KOルートの外へ出ない」制御なので、
既知後退と同じ変更ではない。ただし発火回数が多ければ実質的なcapになるため、
診断値を必須にする。

### 6.2 ターゲット固定のリスク

- Bossを使う前に相手Activeを倒せるようになっても、古い高価値ターゲットへ固執する。
- Hammerで保護が消えた後、優先順が変わる。
- Boss後に対象areaがBenchからActiveへ変わる。

対策:

- serialで同一個体を追う。
- 対象消失、勝利ルート出現、効果防止状態変更時だけ再評価する。
- ターゲット価値リストはターン内固定、ルート可否と必要行動は各MAINで再計算する。

### 6.3 Boss絶対化のリスク

- Activeも既に即勝利なら余計な操作。
- Boss使用後に攻撃不能状態を見落とす。
- 全体保護役をBenchに残したまま別のBasicを呼び、Powerful Handが0になる。
- 3プライズ対象のカードDB判定を名前推測で行う。

対策:

- `_boss_damage_after_spend()`、`_effect_prevented()`、`prize_count()` の既存実装だけを使う。
- `handCount - 1` を別実装しない。
- 「攻撃オプションが提示」「Active攻撃役が実際に支払い可能」もgate条件に入れる。
- カード名文字列でMega/exを推測しない。

### 6.4 汎用探索のリスク

- bitmask atom増加による組合せ爆発。
- 同じ進化対象、同じ手札カード、Supporter権、エネルギー権の二重使用。
- Run Away Drawで盤面唯一ボディを消す。
- Boss用Supporter権とDawn/Hildaを同居させる。
- Hammerを別個体のEnergyに使う。

対策:

- 現行の小さい確定atom集合を維持し、未知ドロー内容や全Trainerは入れない。
- ターゲットごとに最大atom数を制限し、優先上位から到達時点で打ち切る。
- 既存 `fallback_policy.py:1666-1702` のSupporter、進化対象、手札枚数、
  山札枚数制約を保持。
- `needs_boss` ならDawn/Hildaを探索前に除外。
- Hammer atomへ対象serialとenergy serialを持たせる。

## 7. 検証計画

### 7.1 実装を3段階に分ける

| Variant | 変更 |
|---|---|
| A | v19そのまま |
| B | A + 終局Bossゲートだけ |
| C | B + ターゲット必要手札/到達後ドロー停止 |
| D | C + 優先リストと汎用KOルート |

Boss、手札停止、汎用探索を一度に入れない。Bで終局安全性、Cで既知のドロー後退、
Dでプライズ効率を個別に判定する。

### 7.2 静的/golden-state

基準コマンド:

```powershell
.\.venv\Scripts\python.exe -m pytest agents\alakazam\alakazam_ml_v20 -q
```

現行v19は今回の再確認で全テスト成功（88 tests相当の全dot）。
v20では第3.3節のBoss 8状態に加え、次を固定する。

- 70/130/140/210/300/340等の残HPに対する `ceil(hp/20)`。
- ActiveとBoss対象で必要手札がちょうど1枚違う。
- 必要手札到達前はRun Away Drawを使い、到達後は止める。
- Active Dudunsparce退避と唯一ボディ禁止は変更しない。
- Dawn純増+2、Hilda+1、Rare Candy+1、進化純増の一貫性。
- 1位ターゲット到達不能なら2位へ進む。
- BossルートにDawn/Hildaが混ざらない。
- Hammerが選択ターゲットの防止Energyを落とす。
- Fezandipiti exのBench直接攻撃ではBossを要求しない。
- 残り2で自分BenchのFezandipiti exが負債になるrace状態。

### 7.3 local_arena A/B

最低限:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\local_arena.py `
  alakazam_ml_v20 alakazam_ml_v19 --games 500 --seed 1901 --quiet
```

推奨:

- seed 1901/1902、各1,000戦。席順を反転。
- v20 vs v19 mirror。
- ex主体、Mega、単プライズ、Alakazam mirror、Grimmsnarlの固定対面を各500戦以上。
- C（手札停止）は単プライズ長期戦とmirrorを重点監視。
- D（高価値ターゲット）はex/Mega対面を重点監視。

勝率だけでなく、次をvariant別に出す。

- 終局Boss機会 / 即選択 / 同ターン攻撃 / 勝利。
- ターゲット順位、到達不能でskipした回数。
- 1/2/3プライズ別KO数、勝者のKO構成、平均ターン。
- Run Away Draw総数、選択ターゲット致死後の発動数。
- Powerful Hand攻撃時の必要手札超過。
- deckout / no-offense / boardout / fallback / illegal action。
- Fezandipiti exがKOされた回数。

昇格条件案:

- 終局Boss golden-state 100%、公開再生の該当機会100%。
- crash / illegal / policy fallback / observation fallback = 0。
- Cでdeckout率がAより悪化しない。
- Dでvs exの2プライズKO率が上がり、平均KO数または平均ターンが改善。
- 主要対面の勝率を2ポイント超落とさない。
- 総合は最低2,000戦でWilson区間とpaired seed差を確認する。

### 7.4 上位方策一致率

既存コマンド:

```powershell
.\.venv\Scripts\python.exe -X utf8 .\scripts\analyze_alakazam_policy_imitation.py `
  data\runs\alakazam\20260724_top40_current `
  agents\alakazam\alakazam_ml_v20 `
  --ranks 2,3,5,8 `
  --deck-hash cc38cb450b86770a `
  --output data\analysis\v20_imitation_top4.json
```

v19基準は18,749判断で60.8%。ただし終局Bossは上位自身が17/24を見送っているため、
総一致率を唯一の昇格条件にしてはいけない。

比較は次の三分割にする。

1. 意図した不一致: 勝ち確Boss、必要手札到達後の純ドロー停止。
2. 許容する派生不一致: 同じターゲット/同じKOルート内の行動順。
3. 不意の不一致: energy、retreat、保護解除、唯一ボディ安全則。

全体一致率が下がっても1だけなら仕様通り。3が増えたら不採用。

## 8. Claude分析との突き合わせ

この節だけは、上記の独立分析を固定した後に
`data/analysis/v20_logic_analysis.md` を読んで作成した。

### 8.1 同意点

1. **ML再学習は不要。**
   Claude §0と本分析§2は一致する。Boss、attack、Trainer、Energy、Ability、Retreatは
   fallback固定で、変更の主戦場は `fallback_policy.py`。
2. **Bossの部品はあるが絶対ゲートがない。**
   `_boss_target_score()` の `winning` 加点が、PLAY側の `//8` と上限12,000で圧縮され、
   選択全体の不変条件にならないという診断は同じ。
3. **`_winning_gust_ready()` は未接続。**
   dead helperの存在は両分析で確認した。
4. **手札目標がActiveに偏っている。**
   `_achievable_hand()`、`_ko_active_reachable()`、`_active_route_plan()` は
   選択したBench高価値ターゲットを共通目標にしていない。
5. **統合ターゲットリストと任意targetのKOルートが必要。**
   ActiveとBoss/Benchを別々に評価する現状を、
   優先順 → 到達可能性 → ルート行動スコアへ変える構造は一致する。
6. **一律ドローcapは危険。**
   `fallback_policy.py:2373-2387` の既知後退を踏まえ、
   local_arena A/Bを必須とする点も一致する。
7. **実装依存関係は Boss → target/route → target依存hand gate。**
   Claudeの実装順 ①→③→② は妥当。本分析のロールアウトでは、
   Bossの次にまずActive限定の安全な手札停止を単独A/Bし、その後に汎用targetへ広げるが、
   完成形の②は③の選択ターゲットを入力に必要とする。

### 8.2 相違点

#### A. `_winning_gust_ready()` はそのまま再利用しない

Claudeは残サイド2/3を足して再利用する案。本分析は新しい
`_terminal_boss_targets()` を推奨する。

理由:

- `_winning_gust_ready()` はBossが手札にあり合法か、Supporter使用済みでないかを見ない。
- `_active_best_dmg(p)` を使うため、MAINでBossを手札から使う `-1枚 = -20` を反映しない。
- `_boss_damage_after_spend()` はBoss解決中の二重減算回避と効果防止まで既に正しく扱う。

dead helperを結線するより、正しい既存helperを束ねる方が安全。

#### B. 「アンチXerosicバッファ」は入れない

Claude §2は `need_hand + α` までのバッファを例外にする。
本分析は反対する。Xerosicは差分枚数を削るのでなく手札を3枚まで落とすため、
8枚も20枚も被弾後のPowerful Handは原則60。有限の `+α` は防御にならない。

余分に引いたカードが捨てられる量と山札消費は増えるので、対策は
「必要手札到達後に今攻撃する」「再建資源を山札に残す」であり、
目標手札を上乗せすることではない。v19実ログの22被弾、平均10.6枚減もこの判断を支持する。

#### C. 上位のRun Away Draw差の解釈

Claudeはコードコメントの `163 vs 622` をそのまま根拠にし、
上位は総発動が約1/4とする。独立再集計では:

- v19実戦: 90/52 = 1.73回/戦。
- 上位4: 776/261 = 2.97回/戦。

総回数では再現しない。再現するのは
「現在Activeが既に致死なのに発動」の条件付き率
（v19 38.9% vs 上位16.0%）。したがって固定回数や手札capではなく、
ターゲット致死到達後の停止を実装すべき。

#### D. ターゲット価値と到達可能性を分離する

Claude §3の優先度関数は「到達可能性」もターゲットスコアへ入れる。
本分析は、戦略価値リストには入れず、優先順の各個体へ `_ko_route(target)` を順に試す。

到達可能性を価値へ混ぜると、低HP単プライズActiveが常に高価値exより上になりやすく、
ユーザーが求める「最優先を試し、無理なら次点」が再び局所スコアへ戻る。
必要手札は同じ価値内のtie-break程度に留める。

#### E. Bossの絶対性とActive即勝利

Claudeは既存 `_winning_gust_ready()` の「Activeで勝てるならBoss不要」をそのまま採る。
本分析も勝率上はその順を推奨するが、ユーザー原文の字義通りならBossが絶対である。
統合実装はこの例外を暗黙にせず、
「Active terminal attack > terminal Boss」か逆かをgolden testで明示的に固定すべきとする。

#### F. スコア90,000だけでは絶対にならない

ClaudeはBossをterminal級「例: 90,000」に固定する案。
現行にはActive終局90,000、board wipe 95,000があるので、同点順や既存スコアに依存する。
本分析は成立時に他行動を `-1` にするゲート、または既存最大より上の階層定数を推奨する。

### 8.3 Claudeの見落とし

1. **`_winning_gust_ready()` のBoss手札消費漏れと合法性漏れ。**
   「ユーザー要件そのもの」ではなく、現状のままでは境界手札で偽陽性になる。
2. **実クラス名。**
   Claudeと依頼文は `PolicyEngine._score` と書くが、v19実体は
   `AlakazamPolicy._score` (`fallback_policy.py:486`, `:2222`)。
3. **`policy_base.py` とfallbackの重複。**
   `policy_base.py` の `prize_count` を直すだけでは実戦fallbackへ反映されない。
4. **手札差分の不整合。**
   Dawnはroute atomで純増+2だが `_hand_delta()` と `_achievable_hand()` では+1。
   同じ概念の一元化が必要。
5. **routeタイブレークが過剰火力を選ぶこと。**
   現行キー末尾の `-damage` は同手数なら大ダメージを優先する。
6. **BossルートのSupporter排他。**
   Bench targetへBossするルートではDawn/Hildaを同じターンに使えない。
7. **Hammer対象の個体binding。**
   汎用routeでは、別のActiveでなく選択ターゲットの防止Energyを落とす必要がある。
8. **race入力として自分のFezandipiti ex負債。**
   相手残り2サイドで自分Benchに140がいる局面は、同ターン決着の価値を上げる。
9. **52戦の実数。**
   勝ち確Bossは1機会で実行済み、Boss全27回が同ターン攻撃、
   Run Away致死後35回、Xerosic被弾22回、deckout負け3回。
10. **上位も勝ち確Bossを見送る。**
    上位4は24機会中7回だけBoss。要件1は模倣改善ではなく、意図的な戦略差。
11. **過剰火力とKO回数の区別。**
    prizeレポートが示すのは「必要KO回数が多いほど遅い」ことであり、
    攻撃時のダメージoverkill自体が遅いという因果ではない。
12. **具体的なgolden-state、段階A/B、診断昇格条件。**
    特にBoss境界、効果防止、Dawn/Boss排他、ターゲットskipが必要。
13. **Poffinの分類。**
    Poffinは単純な手札ドローではなくBench展開。到達後に抑える場合も、
    Run Away Drawと同じ「純ドロー禁止」ではなく盤面必要性を別評価すべき。

### 8.4 優先順位の違い

Claudeの推奨は **① Boss → ③ target/route → ② hand stop**。
完成形の依存関係として同意する。

本分析はリスクを切り分けるため、実装/昇格を次のように細分化する。

1. **B: 終局Bossゲート。**
   小変更でgolden-stateを100%にする。dead helperは流用せず正しいdamage helperを使う。
2. **C: Active限定の必要手札停止。**
   現行Active routeを使い、Bench Run Away Drawの「既にActive致死」だけを先にA/B。
   52戦ログで発火候補35回があり、効果を早く測れる。
3. **D1: 戦略ターゲット優先リスト。**
   プライズレースとFez負債を含めるが、まだ行動を変えずshadow診断する。
4. **D2: target引数付き確定KOルート。**
   Activeを先に一般化し、次にBench+Boss、最後に他攻撃役へ広げる。
5. **D3: 全ドロー/進化/Trainerを選択ルートへ接続。**
   ここで完成形の要件②を適用する。

Claude案より段階が多い理由は、手札抑制のcabt後退と汎用探索の資源二重使用を
別々に検出するため。最終機能の重要度は同じく ① > ③ > ②だが、
最初の実験価値はBossの次にActive限定Run Away停止が高い。
