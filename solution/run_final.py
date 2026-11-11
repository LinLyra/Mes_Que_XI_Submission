"""Regenerate the exact Mes Que Data XI Phase 1 submission."""
from pathlib import Path
import hashlib
from sequence_v7 import decode_v7
from sequence_v2 import META, RAW, extract, slot_teams, write
from sequence_v3 import SMART, load_events

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission/Mes_Que_Data_XI.csv"

def main():
    dh, da, speed, active = extract(RAW, slot_teams(META))
    states, _, _ = decode_v7(dh, da, speed, active, load_events(SMART))
    write(OUT, states)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("wrote", OUT)
    print("sha256", digest)

if __name__ == "__main__":
    main()
