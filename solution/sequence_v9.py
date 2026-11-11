"""V7 physics/decoding with corrected football semantics for phase priors."""
from pathlib import Path
import numpy as np
from sequence_v2 import META,RAW,emissions,extract,slot_teams,viterbi,write
from sequence_v3 import SMART,load_events
from sequence_v4 import add_control_anchors,duration_cleanup
from semantic_priors import add_semantic_priors

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"evaluation/results/mes_que_data_sequence_v9.csv"
def decode_v9(dh,da,speed,active,events):
    score=add_semantic_priors(emissions(dh,da,speed,active),events)
    score,h,a=add_control_anchors(score,dh,da,weight=1.7)
    return duration_cleanup(viterbi(score,.75),score,np.array([25,25,40])),h,a
def main():
    dh,da,s,a=extract(RAW,slot_teams(META));states,_,_=decode_v9(dh,da,s,a,load_events(SMART));labels=write(OUT,states)
    print('transitions',int(np.sum(labels[1:]!=labels[:-1])),'wrote',OUT)
if __name__=='__main__':main()
