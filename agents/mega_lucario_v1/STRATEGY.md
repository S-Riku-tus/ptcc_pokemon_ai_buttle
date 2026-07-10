# mega_lucario_v1

## 目的

初期の全選択肢共通スコア方式を置き換え、デッキ専用・文脈別の決定的ルールで
対戦を安定させる版です。

## デッキの役割

- Mega Lucario ex：主力アタッカー
- Hariyama：非exアタッカー。Crustle対策
- Solrock：軽量アタッカー
- Riolu / Makuhita：進化元
- Boss's Orders / Switch：攻撃計画に応じた対象・攻撃者の変更
- Fighting Gong / 基本闘エネルギー：攻撃準備
- Carmine / Lillie's Determination：手札展開

## 方策の特徴

- MAIN、初期配置、サーチ、捨て札、交代、対象選択を別々に評価
- ターンごとに攻撃者、攻撃対象、必要エネルギーを計画
- 任意選択で`maxCount`を無条件に埋めない
- 例外時は構造上合法な最小選択へフォールバック
- 同じ盤面では同じ行動を返し、実験の再現性を確保

## 次の改善候補

1. Replayから負けた相手デッキを分類する
2. Crustle、Dragapult、ミラーマッチ別に失敗場面を抽出する
3. エネルギー付与、Boss's Orders、Switchの判断を個別評価する
4. 80試合以上の相手別ベンチマークで変更を判定する
