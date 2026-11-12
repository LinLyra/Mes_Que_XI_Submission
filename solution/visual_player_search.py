"""Search for a missing ball around projected players in ranked intervals."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.load_homography import MATRIX_COLUMNS

VIDEO=ROOT/"data/tactical/BAR-ATH_tactical.mp4"
RAW=ROOT/"data/tactical/BAR-ATH_tactical_atd.txt"
HOM=ROOT/"data/tactical/BAR-ATH_tactical_homography.csv"
CANDIDATES=ROOT/"evaluation/diagnostics/visual_candidates.csv"
OUT=ROOT/"evaluation/diagnostics/player_search.csv"

def flush(model,images,meta,rows,batch):
    if not images:return
    results=model.predict(images,imgsz=160,conf=.15,batch=batch,verbose=False)
    for result,(frame,slot,centre) in zip(results,meta):
        if result.boxes is None or len(result.boxes)==0:continue
        confidence=result.boxes.conf.cpu().numpy();points=result.boxes.xywh[:,:2].cpu().numpy()
        for score,point in zip(confidence,points):
            pixel=centre-80+point
            rows.append((frame,slot,float(score),float(pixel[0]),float(pixel[1])))
    images.clear();meta.clear()

def deduplicate(table,radius=15.):
    kept=[]
    for frame,group in table.groupby("frame"):
        selected=[]
        for row in group.sort_values("confidence",ascending=False).itertuples():
            p=np.array([row.pixel_x,row.pixel_y])
            if all(np.linalg.norm(p-q)>radius for q in selected):
                kept.append(row[1:]);selected.append(p)
    return pd.DataFrame(kept,columns=table.columns)

def add_tracks(table,stride,max_step=80.,minimum=3):
    table=table.copy();table["track_id"]=-1;active={};next_id=0
    for frame in sorted(table.frame.unique()):
        indices=table.index[table.frame==frame].tolist();used=set();new={}
        for track,(last_frame,last_point,length) in active.items():
            if frame-last_frame!=stride:continue
            choices=[]
            for index in indices:
                if index in used:continue
                row=table.loc[index];point=np.array([row.pixel_x,row.pixel_y]);distance=np.linalg.norm(point-last_point)
                if distance<=max_step:choices.append((distance,index,point))
            if choices:
                _,index,point=min(choices);table.loc[index,"track_id"]=track;used.add(index);new[track]=(frame,point,length+1)
        for index in indices:
            if index in used:continue
            row=table.loc[index];point=np.array([row.pixel_x,row.pixel_y]);table.loc[index,"track_id"]=next_id;new[next_id]=(frame,point,1);next_id+=1
        active=new
    lengths=table.groupby("track_id").size();valid=set(lengths[lengths>=minimum].index)
    table["trajectory_valid"]=table.track_id.isin(valid)
    return table

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--top",type=int,default=25);ap.add_argument("--stride",type=int,default=5);ap.add_argument("--confidence",type=float,default=.75);ap.add_argument("--chunk",type=int,default=512);ap.add_argument("--batch",type=int,default=64);ap.add_argument("--model",type=Path,required=True);ap.add_argument("--out",type=Path,default=OUT);a=ap.parse_args()
    from ultralytics import YOLO
    ranked=pd.read_csv(CANDIDATES).head(a.top);target={i for r in ranked.itertuples() for i in range(r.start_frame,r.end_frame,a.stride)}
    H=pd.read_csv(HOM)[MATRIX_COLUMNS].to_numpy(float).reshape(-1,3,3);records={}
    with RAW.open() as handle:
        for line in handle:
            n,players,ball=line.rstrip().split(":");frame=int(n)-1
            if frame not in target or not ball.startswith("N"):continue
            visible=[]
            for slot,text in enumerate(players.split(";")):
                if text.startswith("N"):continue
                q=text.split(",");visible.append((slot,(float(q[0]),float(q[1]))))
            records[frame]=visible
    model=YOLO(a.model);cap=cv2.VideoCapture(str(VIDEO));images=[];meta=[];rows=[];crops=0
    for frame,players in records.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read()
        if not ok:continue
        for slot,xy in players:
            q=H[frame]@np.r_[xy,1.];p=q[:2]/q[2]
            if not(80<=p[0]<1840 and 80<=p[1]<1000):continue
            images.append(cv2.getRectSubPix(image,(160,160),(float(p[0]),float(p[1]))));meta.append((frame,slot,p));crops+=1
            if len(images)>=a.chunk:flush(model,images,meta,rows,a.batch)
    flush(model,images,meta,rows,a.batch);cap.release()
    columns=["frame","player_slot","confidence","pixel_x","pixel_y"]
    table=pd.DataFrame(rows,columns=columns)
    if len(table):
        table=table[table.confidence>=a.confidence];table=deduplicate(table);table=add_tracks(table,a.stride)
    a.out.parent.mkdir(parents=True,exist_ok=True);table.to_csv(a.out,index=False)
    print("missing sampled frames:",len(records),"player crops:",crops)
    print("high-confidence detections:",len(table),"valid trajectory points:",int(table.trajectory_valid.sum()) if len(table) else 0)
    print("wrote:",a.out)

if __name__=="__main__":main()
