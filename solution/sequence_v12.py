"""Experimental v7 plus gated player-centred visual ball trajectories."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from starter.load_homography import MATRIX_COLUMNS
from sequence_v2 import META,RAW,extract,slot_teams,write
from sequence_v3 import SMART,load_events
from sequence_v7 import decode_v7
from visual_ball_flow import fill_distances

TRACKS=ROOT/"evaluation/diagnostics/player_search_top25.csv"
HOM=ROOT/"data/tactical/BAR-ATH_tactical_homography.csv"
OUT=ROOT/"evaluation/results/mes_que_data_player_visual_v12.csv"

def recovered_pitch(tracks,H):
    recovered={}
    for _,group in tracks[tracks.trajectory_valid].groupby("track_id"):
        group=group.sort_values("frame");frames=group.frame.to_numpy(int)
        for frame in range(frames[0],frames[-1]+1):
            pixel=[np.interp(frame,frames,group.pixel_x),np.interp(frame,frames,group.pixel_y),1.]
            value=np.linalg.solve(H[frame],pixel);xy=value[:2]/value[2]
            if np.all(xy>=-.03) and np.all(xy<=1.03):recovered[frame]=xy.astype(np.float32)
    return recovered

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--tracks",type=Path,default=TRACKS);ap.add_argument("--out",type=Path,default=OUT);a=ap.parse_args()
    tracks=pd.read_csv(a.tracks);H=pd.read_csv(HOM)[MATRIX_COLUMNS].to_numpy(float).reshape(-1,3,3);recovered=recovered_pitch(tracks,H)
    teams=slot_teams(META);dh,da,speed,active=extract(RAW,teams);fill_distances(RAW,teams,recovered,dh,da,speed)
    states,_,_=decode_v7(dh,da,speed,active,load_events(SMART));labels=write(a.out,states)
    print("recovered frames:",len(recovered));print("transitions:",int(np.sum(labels[1:]!=labels[:-1])));print("wrote:",a.out)

if __name__=="__main__":main()
