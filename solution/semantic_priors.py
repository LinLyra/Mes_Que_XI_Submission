"""Football-semantic interpretation of automatic Smart Tagging phases."""
import numpy as np

ATTACKING={"BUILD UP","PROGRESSION","BALL IN FINAL THIRD","BALL IN THE BOX",
           "CREATING CHANCES","PLAYERS IN THE BOX","LONG BALL","ATTACKING TRANSITION","GOALS"}
DEFENDING={"DEFENDING IN MIDDLE THIRD","DEFENDING IN DEFENSIVE THIRD",
           "DEFENDING IN ATTACKING THIRD"}

def add_semantic_priors(score,events):
    n=len(score); evidence=np.zeros((n,2),np.float32); count=np.zeros(n,np.float32)
    for code,start,end,tags in events:
        code=code.strip().upper(); team=tags.get("Team")
        side=0 if team=="Barcelona" else 1 if team=="Atletic Club" else None
        if side is None: continue
        if code in DEFENDING: side=1-side
        elif code not in ATTACKING: continue
        start=max(0,start);end=min(n,end)
        if end>start:evidence[start:end,side]+=1;count[start:end]+=1
    covered=count>0
    score[covered,0]+=.22*evidence[covered,0]/count[covered]
    score[covered,1]+=.22*evidence[covered,1]/count[covered]
    return score
