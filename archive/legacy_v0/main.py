# main.py — PTCG AI Battle Challenge 提出用エージェント
import random

# ============================================================
# デッキ(60枚)— カードIDを直接埋め込む
# ※以下のIDは例。自分が実際に使ったIDに置き換えること
# ============================================================
DECK = [
    25, 25, 25, 25,
    27, 27, 27, 27,
    28, 28, 28, 28,
    45, 45, 45, 45,
    67, 67, 67, 67,
] + [1] * 40          # 基本草エネルギー ×40

assert len(DECK) == 60, f"デッキが{len(DECK)}枚です"

# ============================================================
# 行動の優先度(解読済みoption.type対応表)
#   7=ワザ, 8=カードを場に付ける/出す, 3=ポケモン選択,
#   6=エネ選択, 14=ターン終了
# ============================================================
def score(opt):
    t = opt.get("type")
    if t == 7:
        return 100
    if t == 8:
        return 50 + (10 if opt.get("inPlayArea") == 4 else 0)
    if t == 3:
        return 40
    if t == 6:
        return 30
    if t == 14:
        return 0
    return 20

# ============================================================
# エージェント本体(末尾の関数がエントリポイント)
# ============================================================
def agent(obs_dict):
    if obs_dict["select"] is None:
        return DECK
    sel = obs_dict["select"]
    options = sel["option"]
    ranked = sorted(range(len(options)),
                    key=lambda i: score(options[i]) + random.random(),
                    reverse=True)
    return ranked[:sel["maxCount"]]
