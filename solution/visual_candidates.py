"""Rank tactical-video intervals where visual ball recovery may change v7.

This is label-free triage, not manual annotation. It concentrates later video
inference on uncertain state boundaries and missing-ball intervals rather than
running an expensive detector over all 199,319 frames.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sequence_v2 import META, RAW, emissions, extract, slot_teams, viterbi
from sequence_v3 import SMART, add_event_priors, load_events
from sequence_v4 import add_control_anchors, duration_cleanup, runs
from games import GAME, half_video_blocks

OUT = ROOT / "evaluation/diagnostics/visual_candidates.csv"
FPS = 25

def merge_intervals(intervals, gap=25):
    merged=[]
    for start,end in sorted(intervals):
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start,end])
    return merged

def split_intervals(intervals, maximum=500):
    out=[]
    for start,end in intervals:
        while end-start>maximum:
            out.append([start,start+maximum]); start+=maximum
        if end>start: out.append([start,end])
    return out

def main():
    dh,da,speed,active=extract(RAW,slot_teams(META)); events=load_events(SMART)
    score=add_event_priors(emissions(dh,da,speed,active),events)
    score,h,a=add_control_anchors(score,dh,da,weight=1.7)
    states=duration_cleanup(viterbi(score,.75),score,np.array([25,25,40]))
    missing=~(np.isfinite(dh)&np.isfinite(da))
    ordered=np.sort(score,axis=1); margin=ordered[:,-1]-ordered[:,-2]
    transitions=np.flatnonzero(states[1:]!=states[:-1])+1

    graded=np.zeros(len(states),bool)
    for _,first,last in half_video_blocks(GAME): graded[first:last+1]=True
    seed=[]
    # Missing runs near a decoded transition are highest-value visual targets.
    starts,ends=runs(missing.astype(np.uint8))
    for start,end in zip(starts,ends):
        if not missing[start]: continue
        near=np.any((transitions >= start-50)&(transitions <= end+50))
        low=float(np.mean(margin[start:end] < .35)) if end>start else 0
        # Do not turn a minutes-long camera failure into one giant clip. Long
        # runs are represented by transition windows below.
        if (near or low>.35) and end-start<=250:
            seed.append((max(0,start-50),min(len(states),end+50)))
    # Include every transition's +/-2 seconds; merge overlaps afterwards.
    seed.extend((max(0,int(t)-50),min(len(states),int(t)+51)) for t in transitions)
    intervals=split_intervals(merge_intervals(seed))

    restart=np.zeros(len(states),bool)
    for code,start,end,_ in events:
        if code.strip().upper()=="SET PIECES": restart[max(0,start):min(len(states),end)]=True

    rows=[]
    for start,end in intervals:
        valid=graded[start:end]
        if not valid.any(): continue
        local_t=transitions[(transitions>=start)&(transitions<end)]
        miss=float(np.mean(missing[start:end])); low=float(np.mean(margin[start:end]<.35))
        rst=float(np.mean(restart[start:end])); duration=(end-start)/FPS
        priority=3.0*len(local_t)+2.0*miss+low+0.5*rst
        rows.append((priority,start,end,duration,miss,low,rst,len(local_t)))
    rows.sort(reverse=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["rank","start_frame","end_frame","start_seconds",
            "end_seconds","duration_seconds","missing_fraction","low_margin_fraction",
            "restart_fraction","transitions","priority"])
        for rank,(p,s,e,d,m,l,r,t) in enumerate(rows,1):
            w.writerow([rank,s,e,round(s/FPS,2),round(e/FPS,2),round(d,2),
                        round(m,4),round(l,4),round(r,4),t,round(p,4)])
    print("candidate intervals:",len(rows))
    print("candidate video minutes: {:.1f}".format(sum(row[3] for row in rows)/60))
    print("top-50 minutes: {:.1f}".format(sum(row[3] for row in rows[:50])/60))
    print("wrote:",OUT)

if __name__=="__main__": main()
