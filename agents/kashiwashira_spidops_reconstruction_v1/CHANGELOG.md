# CHANGELOG

## 1.0.0 — 2026-07-14

- submission 54603674から60枚を復元。
- submission 54613990との差分（草2→超2）を確認。
- 公開100戦を連続ログ重複除去で再集計。
- 攻撃、進化、Spidopsのトラッシュ草加速、手札からの給エネ、どうぐ対象、カード使用を抽出。
- SETUP / ACCELERATE / PRESSURE / RECOVER / ENDGAMEの状態機械を実装。
- Spidops打点を30×Team Rocket’s Pokémon数として実装。
- Proton、Bug Catching Set、Transceiver、Poké Pad、Ultra Ball、Giovanni、Spidops特性のサブ選択を実装。
- Team Rocket’s Energy→Mewtwo ex、基本草→Tarountula/Spidopsの実測分布を給エネ優先へ反映。
- 勝利攻撃固定、攻撃可能時END禁止、限定的退避、低山札ガードを実装。
- 共通API・合法フォールバック・診断をpolicy_base.pyへ分離。
- 分析再現スクリプトとAPIスタブテストを追加。


## 1.0.1

- Mewtwo exの攻撃を、基礎160＋ベンチエネルギー1枚ごとに60（最大2枚）として復元。
- 追加打点がKOへ変わる場合だけ、必要最小枚数の基本草を捨てる専用選択を実装。
- Team Rocket’s Energyを勝利KO以外では追加コストに使わない。
- Brave Bangle装着Spidopsの対ex +30をKO計算へ反映。
- 分離解決されるMewtwo exのダメージを分析スクリプトで再集計。
- Golden-stateテストを5件から10件へ拡張。
