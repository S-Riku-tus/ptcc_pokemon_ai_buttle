# Git運用方針

## ブランチ

- `main`：提出候補として検証済み
- `feature/<内容>`：個別改善
- `experiment/<内容>`：捨てる可能性のある検証
- `hotfix/<内容>`：提出エラーなど緊急修正

## エージェントの版管理

提出中の`mega_lucario_v1`を直接変更せず、新しいフォルダを作ります。

```bash
python scripts/new_agent.py mega_lucario_v1 mega_lucario_v2
git switch -c feature/mega-lucario-v2
```

改善が失敗した場合でも、`v1`へすぐ戻せます。

## コミット例

```text
feat(agent): improve energy attachment priority
fix(agent): avoid optional over-selection
test(bench): add 100-game Dragapult matchup
docs(strategy): record Crustle loss analysis
chore(build): validate archive root files
```

1コミットに、戦略変更・デッキ変更・ビルド変更を混在させない方が、
勝率変化の原因を追跡しやすくなります。

## タグ

実際にKaggleへ提出した版にはタグを付けます。

```bash
git tag -a submission-2026-07-10-lucario-v1 -m "Kaggle submission"
git push origin submission-2026-07-10-lucario-v1
```

KaggleのSubmission IDとタグ名を`experiments/results.csv`へ記録します。

## Pull Request

チーム開発では、`main.py`変更を直接`main`へpushせず、Pull Requestで次を確認します。

- 60枚のデッキか
- `agent()`が存在するか
- 例外時のフォールバックがあるか
- 先攻・後攻を入れ替えた試験か
- 既存版より統計的に改善しているか
- Replayで新しい致命的行動がないか
