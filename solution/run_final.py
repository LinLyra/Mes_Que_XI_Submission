"""Regenerate the exact Mes Que Data XI Phase 1 submission (v15)."""
from pathlib import Path
import hashlib
from sequence_v2 import write
from sequence_v15 import decode_v15

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission/Mes_Que_Data_XI.csv"

def main():
    states, audit, recovered = decode_v15()
    write(OUT, states)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print("visual tracks fused", len(audit), "recovered frames", len(recovered))
    print("wrote", OUT)
    print("sha256", digest)

if __name__ == "__main__":
    main()
