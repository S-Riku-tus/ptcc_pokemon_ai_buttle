# Marnie's Grimmsnarl ex v1 evaluation

## Decision

上位ログから復元した60枚を維持し、専用の決定論ポリシーを採用する。
短期アブレーションではDawn増量が上振れたが、300戦の確認では差が1ポイントに縮小したため、
公開実戦証拠のないデッキ変更は採用しなかった。

## Implemented routes

- 序盤は盤面が3体になるまでPoffinとBasic展開を優先し、1体盤面での全滅を抑える。
- Punk Upは、現在のGrimmsnarl exを2エネまで満たした後、次のMorgrem/Impidimpへ最大3エネを集中する。
- Adrena-Brainは、Activeを180圏内に入れる対象、30点KO、Shadow Bulletとの30+30ルートを優先する。
- Shadow Bulletのベンチ30点は即KO、次の30点圏内、特性エンジン、複数サイドの順で評価する。
- Shayminによる非ルール持ちのベンチダメージ無効と、Battle Cageによるベンチへのダメカン無効を認識する。
- Boss's OrdersはDark弱点込みの同一ターンKOがある場合に限定する。

## Local battle evidence

ローカルの `vendor/cg` 互換エンジン、先後交互、クラッシュ・不正選択0件。
Kaggle本番のレートを直接予測する数値ではない。

| Opponent | Games | W-L-D | Win rate |
|---|---:|---:|---:|
| generic Grimmsnarl（同一60枚） | 100 | 92-8-0 | 92.0% |
| generic Mega Starmie | 100 | 75-25-0 | 75.0% |
| Alakazam ML v10 | 200 | 97-103-0 | 48.5% |
| Alakazam v2 | 150 | 67-83-0 | 44.7% |
| Alakazam v3 | 150 | 82-68-0 | 54.7% |
| Alakazam ML v7 | 150 | 80-70-0 | 53.3% |
| Alakazam v12 top-sync-full | 150 | 111-39-0 | 74.0% |

別seedの300戦では、復元型がAlakazam ML v10へ139-161（46.3%）。同じ60枚の
generic Grimmsnarlは別200戦で19-181（9.5%）だったため、専用化による改善は明確だった。

100戦の診断runではShadow Bullet 460回、Adrena-Brain 185回、Punk Up検索187回、
攻撃可能なGrimmsnarl exがActiveなのにENDを選んだ回数は0だった。

## Deck ablation

Alakazam ML v10、各100戦、同一seedによる一次スクリーニング。

| List | Wins |
|---|---:|
| 復元型: Petrel 4 / Dawn 1 | 43 |
| Petrel 3 / Dawn 2 | 44 |
| Petrel 2 / Dawn 3 | 55 |
| 4枚目のGrimmsnarl ex | 42 |
| 4枚目のRare Candy | 51 |

上位2候補を別seed・各300戦で確認すると、復元型46.3%、Petrel 2 / Dawn 3型47.3%だった。
差が小さく、他対面および実戦ログでの裏付けもないため、復元型を保持した。

## Packaging

静的検証、専用テスト、共有policyのアーカイブ同梱テストは通過済み。
公式 `cg/` はリポジトリに存在しないため、Kaggle提出tar.gzの実生成には公式competition assetが必要。
