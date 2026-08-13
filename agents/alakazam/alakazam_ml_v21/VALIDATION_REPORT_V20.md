# v20 validation report

検証日: 2026-07-24

## 静的・golden-state

- `py_compile`: pass
- pytest: 109 passed
- 新規テスト: 21

新規テストは以下を固定した。

- 残り2/3サイドでのBoss絶対選択
- Mega exの3プライズ判定
- 残り1/4、Supporter使用済み、攻撃未提示で非発火
- Boss使用後の手札`-1`と効果防止
- `ceil(HP/20)`の境界
- Dawn `+2`
- 高価値ターゲット優先と到達不能時の次点
- BossとDawn/Hildaの排他
- target到達後Run Away Draw停止と、必要時の発動
- 最小overkill tie-break
- Benchの防止Energyに対するHammer個体binding

## 公開上位ログ replay

対象: ranks 2,3,5,8、261戦、18,749判断。

| agent | agreement |
|---|---:|
| v19 | 60.7819% |
| v20 | 60.5526% |

差は`-0.2293pt`。今回の要件が上位ログの一部判断を意図的に上書きするため、
総一致率だけで不採用にはしない。
