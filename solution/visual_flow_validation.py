"""Synthetic-occlusion validation for tactical-video ball optical flow."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.load_homography import MATRIX_COLUMNS

VIDEO=ROOT/"data/tactical/BAR-ATH_tactical.mp4"
RAW=ROOT/"data/tactical/BAR-ATH_tactical_atd.txt"
HOM=ROOT/"data/tactical/BAR-ATH_tactical_homography.csv"

def ball_xy(n):
    xy=np.full((n,2),np.nan,np.float32)
    with RAW.open() as f:
        for line in f:
            fs,_,ball=line.rstrip().split(":")
            if not ball.startswith("N"):
                q=ball.split(",");xy[int(fs)-1]=float(q[0]),float(q[1])
    return xy

def pixels(xy,H):
    p=np.column_stack((xy,np.ones(len(xy))))
    z=np.einsum("nij,nj->ni",H,p)
    return z[:,:2]/z[:,2,None]

def _track(grays,initial):
    point=np.array(initial,np.float32).reshape(1,1,2);result=[point[0,0].copy()]
    params=dict(winSize=(31,31),maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    for old,new in zip(grays,grays[1:]):
        nxt,status,_=cv2.calcOpticalFlowPyrLK(old,new,point,None,**params)
        if status is None or not status[0,0]:return None
        result.append(nxt[0,0].copy());point=nxt
    return np.asarray(result)

def track_window(cap,start,length,initial,final):
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    ok,frame=cap.read()
    if not ok:return None
    grays=[cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)]
    for _ in range(length-1):
        ok,frame=cap.read()
        if not ok:return None
        grays.append(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    forward=_track(grays,initial);backward=_track(grays[::-1],final)
    if forward is None or backward is None:return None
    return forward,backward[::-1],0.5*(forward+backward[::-1])

def main():
    if not VIDEO.exists():raise SystemExit("missing {}".format(VIDEO))
    table=pd.read_csv(HOM);H=table[MATRIX_COLUMNS].to_numpy(float).reshape(-1,3,3)
    xy=ball_xy(len(H));known=np.isfinite(xy[:,0]);pix=pixels(xy,H)
    # Deterministic, widely separated 1-second windows with continuous ATD ball.
    candidates=[]
    for start in range(11185,len(H)-25,250):
        if known[start:start+25].all() and np.all((pix[start:start+25]>5)&
                (pix[start:start+25]<np.array([1915,1075]))):candidates.append(start)
    candidates=candidates[:120];cap=cv2.VideoCapture(str(VIDEO));rows=[]
    for start in candidates:
        tracked=track_window(cap,start,25,pix[start],pix[start+24])
        if tracked is None:continue
        forward,backward,pred=tracked;truth=pix[start:start+25]
        gate=max(np.linalg.norm(forward[-1]-truth[-1]),
                 np.linalg.norm(backward[0]-truth[0]))
        err=np.linalg.norm(pred-truth,axis=1)
        rows.append((float(np.median(err)),float(err[-1]),float(np.max(err)),float(gate)))
    cap.release();a=np.asarray(rows)
    print("windows attempted/succeeded:",len(candidates),len(rows))
    if len(a):
        print("median per-window pixel error: {:.2f}".format(np.median(a[:,0])))
        print("median endpoint error: {:.2f}".format(np.median(a[:,1])))
        print("endpoint <=10px: {:.1%}; <=20px: {:.1%}".format(
            np.mean(a[:,1]<=10),np.mean(a[:,1]<=20)))
        print("p90 endpoint error: {:.2f}".format(np.percentile(a[:,1],90)))
        for gate in (10,20,30):
            accepted=a[:,3]<=gate
            print("gate <= {}px: accepted={:.1%}, median path error={:.2f}, p90={:.2f}".format(
                gate,np.mean(accepted),np.median(a[accepted,0]) if accepted.any() else np.nan,
                np.percentile(a[accepted,0],90) if accepted.any() else np.nan))

if __name__=="__main__":main()
