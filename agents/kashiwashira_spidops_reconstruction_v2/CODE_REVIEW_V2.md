# v2 コード通し監査

## 対象

- `main.py`
- `policy_base.py`
- `deck.csv`
- `tests/test_package.py`

## 確認結果

### デッキ・ロード

- `deck.csv`は60枚。
- v1提出時のdeck snapshotと同一。
- `main.py`は自身と同じディレクトリの`deck.csv`を最優先で読み込む。
- `policy_base.py`をエージェント内へ同梱し、別版の共通ポリシー混入を避ける。

### MAINフェーズの順序

- 勝利KO：`WIN`
- 確定KO：`KO`
- KOを作る行動：`ENABLE_KO`
- 一度限りの安全な検索・ドロー・進化・加速：`PRE_ATTACK`
- 通常攻撃：`ATTACK`

進化を含む通常展開スコアは`KO - 1`以下へ制限しています。統合テストで、非KO盤面ではPoké Pad、KO盤面では攻撃を選ぶことを確認しました。

### Poké Pad

- サポーター使用後も合法。
- `TRANSCEIVER`と処理を分離。
- Tarountula、Spidops、Mimikyu、Articunoの全候補が正の評価。
- 必要な進化ラインに応じてTarountula／Spidopsを最優先。

### Factory・山札

- 手札7枚以上の拒否を削除。
- 山札2枚以下だけ停止。
- 非KO攻撃より先、確定KOより後に使用。

### Giovanni

- Activeの実攻撃可能打点を対象ごとに再計算。
- Mewtwo exの220／280を含める。
- KO不能な対象しかない場合はカード自体を使用しない。
- AlakazamのKOには限定的な追加価値を付与。

### 入れ替え

- Activeとベンチの実打点を比較。
- 現在のKOを失う退避は禁止。
- Mimikyuから準備済みSpidopsへのピボットを許可。
- 退避後の選択も実打点・KO価値中心。

### 対面処理

- Crustle：Mewtwo exのダメージを0として扱う。
- Alakazam：Articuno優先。
- Mega Lucario ex：Mewtwo過剰展開を抑制。
- 高速ex：2本目のTarountula–Spidopsを優先。

### 安全性

- 例外時の合法フォールバックを維持。
- Mewtwo追加コストはKOに必要な最小枚数。
- Team Rocket’s Energyは勝利KO以外で追加コストに使用しない。
- Brave Bangleの+30を攻撃・Giovanni・盤面追加KOの全計算へ反映。

## 自動検証

- `python -m py_compile main.py policy_base.py`：成功
- `python -m compileall`：成功
- API互換スタブテスト：26件成功
- 旧Poké Padバグの文字列・手札7枚Factory停止・非KO Giovanni分岐が残っていないことを静的確認

## 残る制約

実`vendor/cg`と対戦ハーネスがないため、実エンジンでの合法手・カード効果・勝率は未確認です。特に対面分岐のカードIDは添付リプレイ中の実IDを使用していますが、最終的な強さは同一seedのA/B対戦で判断する必要があります。
