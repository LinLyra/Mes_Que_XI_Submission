"""Add player-centred hard negatives to the filtered local-ball dataset."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
DATA=ROOT/"evaluation/diagnostics/ball_detector_filtered"
RAW=ROOT/"data/tactical/BAR-ATH_tactical_atd.txt"
HOM=ROOT/"data/tactical/BAR-ATH_tactical_homography.csv"
VIDEO=ROOT/"data/tactical/BAR-ATH_tactical.mp4"

def project(H,xy):
    q=H@np.r_[xy,1.];return q[:2]/q[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--per-frame",type=int,default=2);ap.add_argument("--minimum-distance",type=float,default=120.);a=ap.parse_args()
    from starter.load_homography import MATRIX_COLUMNS
    H=pd.read_csv(HOM)[MATRIX_COLUMNS].to_numpy(float).reshape(-1,3,3)
    wanted={int(p.stem) for s in ("train","val") for p in (DATA/"images"/s).glob("*.jpg")}
    frames={}
    with RAW.open() as f:
        for line in f:
            n,ps,b=line.rstrip().split(":");i=int(n)-1
            if i not in wanted or b.startswith("N"):continue
            q=b.split(",");ball=(float(q[0]),float(q[1]));players=[]
            for slot,v in enumerate(ps.split(";")):
                if v.startswith("N"):continue
                z=v.split(",");players.append((slot,(float(z[0]),float(z[1]))))
            frames[i]=(ball,players)
    cap=cv2.VideoCapture(str(VIDEO));written={"train":0,"val":0}
    for i,(ball,players) in frames.items():
        split="val" if (i//750)%5==0 else "train";bp=project(H[i],ball);candidates=[]
        for slot,xy in players:
            pp=project(H[i],xy);distance=float(np.linalg.norm(pp-bp))
            if distance>=a.minimum_distance and 80<=pp[0]<1840 and 80<=pp[1]<1000:candidates.append((distance,slot,pp))
        candidates.sort(reverse=True)
        if not candidates:continue
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read()
        if not ok:continue
        for _,slot,pp in candidates[:a.per_frame]:
            stem=f"neg_{i:06d}_{slot:02d}";crop=cv2.getRectSubPix(image,(160,160),(float(pp[0]),float(pp[1])))
            cv2.imwrite(str(DATA/"images"/split/f"{stem}.jpg"),crop,[cv2.IMWRITE_JPEG_QUALITY,92])
            (DATA/"labels"/split/f"{stem}.txt").write_text("");written[split]+=1
    cap.release();print("hard negatives:",written,"total:",sum(written.values()))

if __name__=="__main__":main()
