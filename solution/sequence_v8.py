"""Experimental v7 plus strictly gated, aligned broadcast tracking evidence."""
from pathlib import Path
import numpy as np
from sequence_v2 import META,RAW,extract,slot_teams,write
from sequence_v3 import SMART,load_events
from sequence_v7 import decode_v7
from dual_source import extract_broadcast,fuse

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"evaluation/results/mes_que_data_sequence_v8.csv"

def main():
    dh,da,speed,active=extract(RAW,slot_teams(META))
    bh,ba,bactive=extract_broadcast(); fh,fa,used=fuse(dh,da,bh,ba,bactive)
    states,_,_=decode_v7(fh,fa,speed,active,load_events(SMART)); labels=write(OUT,states)
    print("broadcast frames fused:",int(used.sum()))
    print("transitions:",int(np.sum(labels[1:]!=labels[:-1])))
    print("wrote:",OUT)

if __name__=="__main__": main()
