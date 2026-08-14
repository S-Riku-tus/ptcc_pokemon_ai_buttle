import json
import sys

data = json.load(open("experiments/grimmsnarl_ml_v24/v24_verdict.json", encoding="utf-8"))


def line(label, b):
    if not isinstance(b, dict) or "games" not in b:
        return f"  {label}: {b}"
    if b["games"] == 0:
        return f"  {label}: 0"
    w = b["wilson95"]
    return (
        f"  {label:<28} n={b['games']:>3}  {b['record']:>7}  "
        f"{b['win_rate']:.3f}  [{w[0]:.3f},{w[1]:.3f}]  opp {b['opp_mean']}"
    )


for section in sys.argv[1:]:
    print(f"=== {section} ===")
    node = data[section]
    for key, value in node.items():
        if isinstance(value, dict) and "games" in value:
            print(line(key, value))
        elif isinstance(value, dict):
            print(f" [{key}]")
            for k2, v2 in value.items():
                if isinstance(v2, dict) and "games" in v2:
                    print(line(k2, v2))
                else:
                    print(f"  {k2}: {json.dumps(v2, ensure_ascii=False)}")
        else:
            print(f"  {key}: {value}")
    print()
