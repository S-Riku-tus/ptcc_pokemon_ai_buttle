# CHANGELOG

## 2.0.0 — 2026-07-14

### P0修正

- Poké Padの`TO_HAND`候補を`supporter_search_value()`へ渡していたバグを修正し、専用`poke_pad_value()`を追加。
- Poké Padをサポーター使用後に禁止していた条件を削除。
- Poké Pad候補のTarountula／Spidops／Mimikyu／Articunoをすべて正の値で評価し、空選択を防止。

### ターン内行動順序

- `PRE_ATTACK` Tierを追加し、Poké Pad、Factory、Bug Catching Set、Transceiver、Ariana、Lillie、進化、Spidops加速を通常の非KO攻撃より先に処理。
- 確定KOと勝利KOは`PRE_ATTACK`より上位を維持。
- 進化スコアを`KO`未満へ上限設定し、確定KOを取り逃さないことを保証。
- Brave Bangle装着時の盤面追加によるKO計算へ+30を反映。

### ドロー・検索

- 山札フロアを「残りサイド+1」から2枚へ変更。
- Factoryの手札7枚以上拒否を削除し、非KO攻撃前に高優先で使用。
- Transceiverは、初動不足時のみProtonを最優先、展開後はArianaを基本優先。
- Protonの盤面5体未満固定優先を削除。

### 攻撃・入れ替え

- Giovanniを同ターンKOまたは勝利KOが成立する場合に限定。
- Mewtwo exのGiovanni打点計算へ220/280の必要最小追加打点を反映。
- 初期ActiveをMimikyu > Tarountula > Articuno > Mewtwo exへ変更。
- Activeとベンチの実打点を比較する退避ロジックを実装し、MimikyuからSpidopsへのピボットを復元。
- 退避後のActive選択を、HPではなく実際の攻撃可能打点とKO価値中心へ変更。

### 対面処理

- CrustleにはMewtwo exの攻撃ダメージを0として扱い、Mewtwoの展開・給エネを抑制。
- Alakazamが見える場合、最初のArticunoを優先し、KO可能なAlakazamへのGiovanni価値を加算。
- Mega Lucario exではMewtwo exの過剰展開を抑制。
- Mega Lucario ex／Cinderace／Archaludon ex／Mega Starmie exでは2本目のTarountula–Spidopsラインを優先。

### 検証

- API互換スタブのテストを10件から26件へ拡張。
- Poké Pad、Factory、Giovanni、Transceiver、Mimikyuピボット、KO優先、Crustle、Alakazam、高速exをGolden-state化。

## 1.0.1

- Mewtwo exの160/220/280と必要最小の基本草選択を実装。
- Team Rocket’s Energyを勝利KO以外では追加コストに使わない。
- Brave Bangleの対ex +30を反映。

## 1.0.0

- submission 54603674の60枚と公開100戦から初回クリーンルーム再構成を作成。
