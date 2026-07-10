# vendor/

## vendor/cg — ローカル対戦用エンジン＋互換API（Git管理外）

`vendor/cg/` はローカルベンチマーク（`scripts/local_arena.py`）専用です。

- `game.py` / `sim.py` / `libcg.so`（Linux用） / `cg.dll`（Windows用）:
  公式ラダーと同一バージョンの `kaggle-environments==1.30.1` wheel の
  `kaggle_environments/envs/cabt/cg/` からコピーしたcabtエンジン
- `cards.json` / `attacks.json`: エンジンの `AllCard` / `AllAttack` エクスポートから
  抽出した全1267枚のカードデータ
- `api.py`: 公式 `cg/api.py` のローカル互換シム（自作）。**提出禁止**
  （`scripts/build_submission.py` は "compatibility shim" マーカーを検出して
  自動的にスキップします）

Kaggle提出時は、Simulationコンペデータの公式 `sample_submission/cg` が
自動検出されてバンドルされます。

## 再生成方法

```bash
pip download kaggle-environments==1.30.1 --no-deps --python-version 3.11 \
  --only-binary=:all: -d /tmp/ke
cd /tmp/ke && unzip kaggle_environments-*.whl -d kex
cp kex/kaggle_environments/envs/cabt/cg/{game.py,sim.py,libcg.so,cg.dll} vendor/cg/
```

カードデータの再抽出:

```python
import ctypes, json
lib = ctypes.cdll.LoadLibrary("vendor/cg/libcg.so")  # Windowsは cg.dll
lib.GameInitialize()
lib.AllCard.restype = ctypes.c_char_p
lib.AllAttack.restype = ctypes.c_char_p
json.dump(json.loads(lib.AllCard().decode()), open("vendor/cg/cards.json", "w"))
json.dump(json.loads(lib.AllAttack().decode()), open("vendor/cg/attacks.json", "w"))
```
