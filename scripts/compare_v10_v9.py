"""Head-to-head comparison harness for alakazam741_v10_route_eta vs alakazam741_v9_top8_core.

Runs alternating-seat games and, for whichever agent is the PRIMARY, reconstructs the requested
metrics from its own MAIN decisions by rebuilding its policy object (reusing its real
damage/state/backup predicates — no metric re-implementation drift):
  * T2 attack rate (attacked by the 2nd own turn)
  * overall own-turn attack rate
  * Alakazam attacks per game
  * board-collapse losses (lost games whose min own in-play count fell to <=1)
  * no-line endings (games ending with no Abra/Kadabra/Alakazam presence at the last own decision)
  * energy-stalled turns (own turns with no attack while energy-starved / backup fuel short)
  * non-mirror Xerosic uses (chosen Xerosic play while the opponent is not a mirror)
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor")); sys.path.insert(0, str(ROOT))
from scripts.agent_loader import load_dir_agent_module
from cg.api import OptionType, SelectContext, to_observation_class
from cg.game import battle_start, battle_select, battle_finish

ALAKAZAM_LINE = (741, 742, 743)  # Abra, Kadabra, Alakazam
XEROSIC = 1197

def load(name):
    m = load_dir_agent_module((ROOT / "agents" / name).resolve())
    return m.agent, m

class Rec:
    def __init__(self, M): self.M=M; self.reset()
    def reset(self):
        self.turns={}; self._seen=[]; self.min_board=None
        self.last_has_line=True; self.energy_stall=0; self.nonmirror_xerosic=0
    def _oti(self, tn):
        if tn not in self._seen: self._seen.append(tn)
        return self._seen.index(tn)+1
    def observe(self, obs_dict, action):
        sel = obs_dict.get("select") if isinstance(obs_dict, dict) else None
        if sel is None or not action: return
        try: obs = to_observation_class(obs_dict)
        except Exception: return
        if obs.current is None or obs.select is None: return
        if obs.select.context != SelectContext.MAIN: return
        opts = obs.select.option or []
        idx = action[0] if isinstance(action,(list,tuple)) and action else None
        if idx is None or not (0<=idx<len(opts)): return
        opt = opts[idx]
        try: pol = self.M.AlakazamPolicy(obs)
        except Exception: return
        me = obs.current.players[obs.current.yourIndex]
        board = sum(p is not None for p in (me.active+me.bench))
        self.min_board = board if self.min_board is None else min(self.min_board, board)
        # line presence at this (latest) own decision
        line_present = any(pol.field[c] for c in ALAKAZAM_LINE) or any(pol.hand[c] for c in ALAKAZAM_LINE)
        self.last_has_line = bool(line_present)
        ti = self._oti(obs.current.turn)
        rec = self.turns.setdefault(ti, {"attacked":False,"alakazam":False,"stall_flag":False})
        t = opt.type
        if t == OptionType.ATTACK:
            dmg = pol._attack_damage_for_option(opt)
            if dmg>0:
                rec["attacked"]=True
                a = me.active[0] if me.active else None
                if a is not None and a.id==743: rec["alakazam"]=True
        elif t == OptionType.PLAY:
            if pol._play_card_id(opt)==XEROSIC:
                opp_board=[p for p in (obs.current.players[1-obs.current.yourIndex].active+
                                       obs.current.players[1-obs.current.yourIndex].bench) if p is not None]
                if not any(p.id in ALAKAZAM_LINE for p in opp_board):
                    self.nonmirror_xerosic += 1
        # energy-stall signal (VERSION-AGNOSTIC: computed from raw board/hand, not policy methods):
        # a line body (Kadabra/Alakazam) in play has no energy AND no psychic energy is in hand.
        needs_fuel = any((b is not None and b.id in (742,743) and len(getattr(b,"energies",[]) or [])==0)
                         for b in (me.active+me.bench))
        psychic_in_hand = (pol.hand.get(5,0)+pol.hand.get(19,0))>0
        if needs_fuel and not psychic_in_hand:
            rec["stall_flag"]=True
    def summarize(self, won, lost):
        own=len(self.turns); atk=sum(v["attacked"] for v in self.turns.values())
        alak=sum(v["alakazam"] for v in self.turns.values())
        by2=any(self.turns[t]["attacked"] for t in self.turns if t<=2)
        # energy-stalled turns = turns flagged energy-short AND no attack happened
        stall=sum(1 for v in self.turns.values() if v["stall_flag"] and not v["attacked"])
        return {"own":own,"atk":atk,"alak":alak,"by2":by2,"won":won,"lost":lost,
                "board_collapse": bool(lost and (self.min_board is not None and self.min_board<=1)),
                "no_line_end": (not self.last_has_line),
                "energy_stall":stall,"nonmirror_xerosic":self.nonmirror_xerosic}

def play(agents, decks, pseat, rec, stats, max_steps=8000):
    obs, start = battle_start(decks[0], decks[1])
    if obs is None: raise RuntimeError("battle_start failed")
    try:
        for _ in range(max_steps):
            cur=obs["current"]
            if cur["result"]>=0: return 0 if cur["result"]==0 else 1 if cur["result"]==1 else -1
            seat=cur["yourIndex"]
            try: action=agents[seat](obs)
            except Exception: stats["crash"][seat]+=1; return 1-seat
            if seat==pseat:
                try: rec.observe(obs, action)
                except Exception: pass
            try: obs=battle_select(list(action))
            except Exception: stats["illegal"][seat]+=1; return 1-seat
        return -1
    finally: battle_finish()

def run(primary, opp, games, seed):
    pa, pm = load(primary); oa, om = load(opp)
    pdeck=pa({"select":None}); odeck=oa({"select":None})
    random.seed(seed)
    per=[]; stats={"crash":[0,0],"illegal":[0,0]}; rec=Rec(pm); pw=0
    for g in range(games):
        pf=(g%2==0)
        agents=[pa,oa] if pf else [oa,pa]; decks=[pdeck,odeck] if pf else [odeck,pdeck]
        ps=0 if pf else 1; rec.reset()
        r=play(agents,decks,ps,rec,stats)
        won=(r==ps); lost=(r!=-1 and r!=ps); pw+=int(won)
        per.append(rec.summarize(won,lost))
    n=len(per); to=sum(x["own"] for x in per); ta=sum(x["atk"] for x in per)
    return {"primary":primary,"opp":opp,"games":n,"win_rate":pw/n,
            "T2_attack_rate":sum(x["by2"] for x in per)/n,
            "overall_own_turn_attack_rate": (ta/to) if to else 0.0,
            "alakazam_attacks_per_game": sum(x["alak"] for x in per)/n,
            "board_collapse_losses": sum(x["board_collapse"] for x in per),
            "no_line_endings": sum(x["no_line_end"] for x in per),
            "energy_stalled_turns_total": sum(x["energy_stall"] for x in per),
            "nonmirror_xerosic_uses_total": sum(x["nonmirror_xerosic"] for x in per),
            "crashes": sum(stats["crash"]), "illegal": sum(stats["illegal"])}

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--games",type=int,default=120); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out",default=None); a=ap.parse_args()
    res={"v10_as_primary":run("alakazam741_v10_route_eta","alakazam741_v9_top8_core",a.games,a.seed),
         "v9_as_primary":run("alakazam741_v9_top8_core","alakazam741_v10_route_eta",a.games,a.seed)}
    print(json.dumps(res,ensure_ascii=False,indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
        print("wrote",a.out)
