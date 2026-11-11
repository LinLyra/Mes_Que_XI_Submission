"""High-confidence broadcast ATD evidence aligned to tactical frames."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/broadcast/BAR-ATH_broadcast_atd.txt"
META = ROOT / "data/broadcast/BAR-ATH_broadcast_atd.xml"
N = 164488
ALIGNMENTS = ((11185, 90652, 4014), (110808, 190198, 26920))

def extract_broadcast():
    root = ET.parse(META).getroot(); home = next(root.iter("Team")).get("id")
    teams = np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")])
    dh=np.full(N,np.nan,np.float32); da=dh.copy(); active=np.zeros(N,np.uint8)
    with RAW.open() as handle:
        for line in handle:
            fs, ps, bs = line.rstrip().split(":"); i=int(fs)-1; players=ps.split(";")
            active[i]=sum(not p.startswith("N") for p in players)
            if bs.startswith("N"): continue
            z=bs.split(","); bx,by=float(z[0]),float(z[1]); best=[np.inf,np.inf]
            for slot,p in enumerate(players):
                if p.startswith("N"): continue
                q=p.split(","); d=np.hypot((float(q[0])-bx)*105,(float(q[1])-by)*68)
                best[int(teams[slot])]=min(best[int(teams[slot])],d)
            if np.isfinite(best[0]): dh[i]=best[0]
            if np.isfinite(best[1]): da[i]=best[1]
    return dh,da,active

def fuse(tactical_h,tactical_a,bh,ba,active):
    h=tactical_h.copy(); a=tactical_a.copy(); used=np.zeros(len(h),bool)
    for lo,hi,offset in ALIGNMENTS:
        ti=np.arange(lo,hi); bi=ti-offset
        trusted=(np.isfinite(bh[bi])&np.isfinite(ba[bi])&(active[bi]>=18)&
                 (np.minimum(bh[bi],ba[bi])<2.0)&(np.abs(bh[bi]-ba[bi])>5.0))
        missing=~(np.isfinite(h[ti])&np.isfinite(a[ti])); take=trusted&missing
        h[ti[take]]=bh[bi[take]]; a[ti[take]]=ba[bi[take]]; used[ti[take]]=True
    return h,a,used
