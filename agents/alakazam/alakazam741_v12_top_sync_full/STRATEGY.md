# alakazam741_v12_top_sync 戦略

## 根拠と狙い

現在1位のsubmission `54662660`について、2026-07-15取得の公開アーカイブ164 replayを確認し、
serial-to-card対応から60枚を再構成した。トップ版のソースコードは使用せず、公開対戦ログで確認できる
カード構成と行動傾向をv11の状態分類・攻撃予約・`backup_eta`へ組み込んだ。

同チームの既存公開ログ統合集計99試合では、平均初攻撃2.31自ターン、攻撃ターン率64.5%、
フーディン攻撃5.24回/試合、ノココッチ循環5.63回/試合、フーディン攻撃時手札14.17枚だった。
Boss使用ターンの攻撃接続は76.25%である。v12はこの「循環を行うが攻撃を最後に確定する」形を採用し、
v11の過剰手札と山札切れを抑える。

## デッキ

カードIDはエンジンの現行カード表と公開replayで照合した。

- Enriching Energy: `13`。Colorless 1個を提供し、手札から付けると4枚引くACE SPEC。
- Boss's Orders: `1182`。
- Nighttime Mine: `1266`。Tera Pokémonの攻撃コストをColorless 1個増やすStadium。

60枚、同名4枚制限内、ACE SPECはID 13の1枚だけである。

## ターン進行

### 最初の自ターン

優先順位は、Abra本体3体、Dunsparce 1体、攻撃用Psychic Energyまたは進化札、即時脅威への
Shaymin、明確な役割を持つFezandipiti exの順とする。3体のAbraは3本の完成経路とは数えない。
Basic検索の一括選択もこの役割順で止まり、空き枠を無条件に埋めない。

### 初回攻撃前

場に進化可能なAbra、手札にAlakazamとRare Candy、当該ラインのエネルギーまたは今ターンの
Psychic手貼りがある場合、最初のAlakazamはCandyを優先する。今ターン攻撃できない、進化時ドローが
必要、既に攻撃可能Alakazamがいる、または後続作成ではKadabraを優先する。

### 攻撃開始後

Active KOと`backup_eta <= 1`が同時に成立したら、任意検索・ドローを攻撃より下へ置く。
必要手札枚数は`ceil(Active HP / 20)`で計算する。追加行動を許すのは、現在KO、勝利KO、
次ターン後続、Boss KO、ロック解除、または循環による山札維持へ直接つながる場合だけである。

## Enriching EnergyとDudunsparce

主対象はDunsparceである。ただし同じ手貼りでAlakazamラインへPsychicを付けて初回攻撃を作れる場合、
攻撃完成を優先する。Enriching付きDudunsparceのRun Away Drawには追加点を与え、進化元・ACEを山札へ
戻す。唯一のDudunsparceを循環する前に別Dunsparceを置ける状態では、引継ぎ本体を先に置く。
循環後はDunsparce本体不足を新しい役割不足として検索・再展開する。最後の場の1体を消す能力使用は
従来どおり禁止する。

## Fezandipiti ex

- `DRAW_ONLY`: 直前の相手ターンに自分のPokémonがKOされ、手札10枚未満、山札余裕あり、現在KO未達
  のときだけ特性を使う。このモードでは給エネしない。
- `ALTERNATE_ATTACKER`: Powerful Handが攻撃効果保護で通らず、Hammerの方が速くなく、100 damageで
  勝利・複数Prize・保護役・主攻撃役・希少進化を倒せる場合だけ使う。手札のEnergy、Hilda、または
  Night Stretcherで3 Energy完成を説明できる場合だけ給エネする。

## Boss、Hammer、Xerosic

Bossは同ターン攻撃とKOを必須とする。対象はHP順ではなく、Prize、保護役、主攻撃役、進化重要度、
現Activeとの差、Boss 1枚消費後のPowerful Handを合成して選ぶ。Boss解決後は攻撃以外を禁止する。

Hammerは、攻撃効果保護を外す、Activeを止める、特殊効果を消す、現KOと合わせて後続Energyを削る、
のいずれかを要求する。2 Energy固定条件は使わず、1 Energy不足でも固有効果または実攻撃停止があれば
候補にする。ただし現在攻撃・KOは失わない。

非ミラーXerosicは、意味ある攻撃、使用後打点維持、相手手札6枚以上、手札依存盤面、Bossより高価値、
山札余裕、非`LOCKED`をすべて要求する。攻撃なしのSupporter単独ENDは行わない。

## ShayminとNighttime Mine

Shayminは相手Activeのbench-damage attackだけを見る。現在支払えるか次の1枚で支払え、実際にAbra系・
Dunsparce系がKO圏で、Flower Curtainがそのattack damageを防ぎ、Abra 3体とDunsparce 1体の枠を壊さず、
現在攻撃も失わない場合だけ置く。ベンチ上の将来候補だけでは置かない。

Nighttime Mineは、Tera Activeの現在支払えるattackを追加Colorlessで止める場合、または完成した相手の
有利Stadiumを剥がす場合に使う。現在KO、勝利KO、Powerful Hand打点、山札残量を先に検査する。

## 維持する不変条件

- 攻撃可能時にENDしない。
- 0 damageのPowerful Handを選ばない。
- 攻撃可能Alakazamを盾目的で退避させない。
- 意味のない手動退避をしない。
- 手札消費で現在KOまたは意味あるattackを消さない。
- `backup_eta`、Night Stretcher仮想評価、PrizeTracker、合法フォールバックを維持する。
- 起動時に60枚、未知ID、4枚制限、ACE SPEC合計を検査する。

