"""Generate algorithmic ball-image pseudo-labels from ATD and homography.

No human labels or public possession GT are read. The output is a private,
derived manifest and is therefore written under evaluation/diagnostics/.
"""
from __future__ import annotations

import csv, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from games import GAME, half_video_blocks
from starter.load_homography import MATRIX_COLUMNS

RAW=ROOT/"data/tactical/BAR-ATH_tactical_atd.txt"
HOM=ROOT/"data/tactical/BAR-ATH_tactical_homography.csv"
OUT=ROOT/"evaluation/diagnostics/visual_pseudolabels.csv"
WIDTH,HEIGHT=1920,1080

def ball_track(n):
    x=np.full(n,np.nan,np.float32);y=x.copy();speed=x.copy()
    with RAW.open() as f:
        for line in f:
            fs,_,ball=line.rstrip().split(":");i=int(fs)-1
            if ball.startswith("N"):continue
            v=ball.split(",");x[i],y[i]=float(v[0]),float(v[1])
            try:speed[i]=float(v[2])
            except ValueError:pass
    return x,y,speed

def main():
    table=pd.read_csv(HOM);n=len(table);x,y,speed=ball_track(n)
    H=table[MATRIX_COLUMNS].to_numpy(float).reshape(-1,3,3)
    valid=table.valid.to_numpy(bool)&np.isfinite(x)&(table.reprojection_error.to_numpy()<2.0)
    play=np.zeros(n,bool)
    for _,first,last in half_video_blocks(GAME):play[first:last+1]=True
    valid &= play
    frames=np.flatnonzero(valid)[::5]  # 5 fps is ample for highly correlated video.
    points=np.column_stack((x[frames],y[frames],np.ones(len(frames))))
    pixels=np.einsum("nij,nj->ni",H[frames],points)
    px,py=pixels[:,0]/pixels[:,2],pixels[:,1]/pixels[:,2]
    inside=(px>=32)&(px<WIDTH-32)&(py>=32)&(py<HEIGHT-32)
    frames,px,py=frames[inside],px[inside],py[inside]

    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["frame","pixel_x","pixel_y","crop_radius",
                                    "ball_speed","split"])
        for frame,cx,cy in zip(frames,px,py):
            # Blocked temporal split prevents adjacent frames leaking between
            # detector training and validation.
            split="val" if (int(frame)//750)%5==0 else "train"
            w.writerow([int(frame),round(float(cx),2),round(float(cy),2),32,
                        round(float(speed[frame]),3) if np.isfinite(speed[frame]) else "",split])
    train=sum((frames//750)%5!=0);val=len(frames)-train
    print("pseudo-labelled frames:",len(frames),"train:",train,"val:",val)
    print("coverage of visible in-play ball: {:.1%}".format(len(frames)*5/max(1,valid.sum())))
    print("wrote:",OUT)

if __name__=="__main__":main()
