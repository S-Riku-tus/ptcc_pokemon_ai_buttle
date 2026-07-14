"""Non-mirror comparison: run v10 and v9 (each as primary) vs generic non-Alakazam opponents,
reusing the Rec/play harness from compare_v10_v9 to capture the same metrics — the point is the
non-mirror Xerosic-use count and that attack tempo holds against a genuinely different deck."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor")); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "agents" / "_base"))
import scripts.compare_v10_v9 as C
from generic_policy import make_generic_agent

def load_generic(name):
    deck=[int(x) for x in (ROOT/"agents"/"_opponents"/name/"deck.csv").read_text(encoding="utf-8-sig").split()]
    return make_generic_agent(deck), deck

def run(primary, opp_name, games, seed):
    pa, pm = C.load(primary); oa, odeck = load_generic(opp_name); pdeck=pa({"select":None})
    random.seed(seed)
    per=[]; stats={"crash":[0,0],"illegal":[0,0]}; rec=C.Rec(pm); pw=0
    for g in range(games):
        pf=(g%2==0)
        agents=[pa,oa] if pf else [oa,pa]; decks=[pdeck,odeck] if pf else [odeck,pdeck]
        ps=0 if pf else 1; rec.reset()
        r=C.play(agents,decks,ps,rec,stats)
        won=(r==ps); lost=(r!=-1 and r!=ps); pw+=int(won)
        per.append(rec.summarize(won,lost))
    n=len(per); to=sum(x["own"] for x in per); ta=sum(x["atk"] for x in per)
    return {"primary":primary,"opp":opp_name,"games":n,"win_rate":pw/n,
            "T2_attack_rate":sum(x["by2"] for x in per)/n,
            "overall_own_turn_attack_rate":(ta/to) if to else 0.0,
            "alakazam_attacks_per_game":sum(x["alak"] for x in per)/n,
            "board_collapse_losses":sum(x["board_collapse"] for x in per),
            "no_line_endings":sum(x["no_line_end"] for x in per),
            "energy_stalled_turns_total":sum(x["energy_stall"] for x in per),
            "nonmirror_xerosic_uses_total":sum(x["nonmirror_xerosic"] for x in per),
            "crashes":sum(stats["crash"]),"illegal":sum(stats["illegal"])}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--games",type=int,default=60)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--opps",default="grimmsnarl,megastarmie")
    ap.add_argument("--out",default=None); a=ap.parse_args()
    res={}
    for opp in a.opps.split(","):
        res[f"v10_vs_{opp}"]=run("alakazam741_v10_route_eta",opp,a.games,a.seed)
        res[f"v9_vs_{opp}"]=run("alakazam741_v9_top8_core",opp,a.games,a.seed)
    print(json.dumps(res,ensure_ascii=False,indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8"); print("wrote",a.out)
