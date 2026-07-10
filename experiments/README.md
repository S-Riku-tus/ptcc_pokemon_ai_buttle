# Experiments

`results.csv`は、エージェント変更と対戦結果の対応を残すための台帳です。

`commit_sha`には試験時のGit commitを記録します。

```bash
git rev-parse --short HEAD
```

同じコードでも試合数が少ないと結果が変動するため、試合数と対戦相手を必ず残します。
