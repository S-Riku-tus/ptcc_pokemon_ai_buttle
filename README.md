# PTCG AI Battle Challenge Agent

ポケモンカードゲーム AI Battle Challenge用のGit管理リポジトリです。

この構成では、提出中の安定版を直接上書きせず、`agents/`配下に戦略・バージョンごとの
エージェントを保存します。Kaggleへ提出する`submission.tar.gz`、取得したReplay JSON、
実験途中のログはGit管理対象から分離します。

## ディレクトリ構成

```text
ptcg-ai-battle/
├── agents/
│   └── mega_lucario_v1/       # 現在の改善版エージェント
│       ├── main.py
│       ├── deck.csv
│       ├── metadata.json
│       └── STRATEGY.md
├── archive/
│   └── legacy_v0/             # 初期の20連敗版。比較用で提出非推奨
├── scripts/
│   ├── build_submission.py    # submission.tar.gz生成
│   ├── validate_agent.py      # 静的検証
│   ├── benchmark.py           # ローカル対戦
│   ├── fetch_replays.py       # Kaggle Episode取得
│   └── new_agent.py           # 新バージョン複製
├── kaggle/
│   └── create_submission_from_git.py
├── data/
│   ├── replays/               # Replay JSON。原則Gitに入れない
│   ├── logs/                  # Agent logs。原則Gitに入れない
│   └── summaries/             # 集計CSVなどはGit管理可能
├── experiments/
│   └── results.csv            # 実験結果一覧
├── docs/
├── tests/
└── artifacts/                 # submission.tar.gz等。Gitに入れない
```

## 最初のGit登録

```bash
git init
git add .
git commit -m "chore: initialize PTCG agent repository"
git branch -M main
git remote add origin <作成したGitHubリポジトリURL>
git push -u origin main
```

`main`ブランチには、提出可能で検証済みの状態だけを置く運用を推奨します。
改善作業は次のようなブランチで行います。

```bash
git switch -c feature/lucario-energy-policy
```

## 現在版の検証

```bash
python scripts/validate_agent.py --agent mega_lucario_v1
```

## Kaggle提出物の生成

Kaggle NotebookでSimulation Competition DataをInputに追加した状態なら、次で生成できます。

```bash
python scripts/build_submission.py \
  --agent mega_lucario_v1 \
  --output /kaggle/working/submission.tar.gz
```

公式`cg/`を自動検出できない場合：

```bash
python scripts/build_submission.py \
  --agent mega_lucario_v1 \
  --cg-source /kaggle/input/.../sample_submission/cg \
  --output /kaggle/working/submission.tar.gz
```

## 新しいエージェント版を作る

既存版を直接上書きせず、複製してから変更します。

```bash
python scripts/new_agent.py mega_lucario_v1 mega_lucario_v2
```

生成後：

```text
agents/mega_lucario_v2/
```

の`main.py`、`deck.csv`、`STRATEGY.md`を変更してください。

## 対戦履歴の取得

Kaggle API認証済みの環境では、Submission IDからEpisode一覧とReplayを取得できます。

```bash
python scripts/fetch_replays.py \
  --submission 12345678 \
  --output data/replays/submission_12345678
```

Episode IDが分かっている場合は認証なしの公開CDN取得も可能です。

```bash
python scripts/fetch_replays.py \
  --episode 80411394 80408508 \
  --output data/replays/manual
```

## 実験の記録

`experiments/results.csv`に最低限、次を残します。

- エージェント名
- Git commit SHA
- 対戦相手
- 試合数
- 勝敗
- エラー数
- 変更内容
- 備考

勝率だけでなく、先攻・後攻、相手デッキ別、エラー数も分けて記録します。

## Gitに入れないもの

次は`.gitignore`で除外しています。

- `submission.tar.gz`
- `cg/`と`vendor/cg/`
- Replay JSON
- Agent logs
- Kaggle APIキー
- 一時ファイル、キャッシュ

学習済みモデルを将来保存する場合、容量が大きければGit LFSを利用してください。