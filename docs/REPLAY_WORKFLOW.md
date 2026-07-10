# Replay取得・保管

Replay JSONは容量が増えるため、原則としてGitへ直接コミットしません。

```text
data/replays/
└── submission_<ID>/
    ├── episode_<ID>.json
    └── ...
```

取得例：

```bash
python scripts/fetch_replays.py \
  --submission <SUBMISSION_ID> \
  --output data/replays/submission_<SUBMISSION_ID>
```

分析後は、生JSONではなく次をGit管理します。

- 集計CSV
- 失敗パターンのMarkdown
- 修正に使ったEpisode ID一覧
- 代表Replayへの説明

秘密情報を含む`kaggle.json`やAPIキーは絶対にコミットしません。
