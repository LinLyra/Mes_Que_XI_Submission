"""Regenerate the exact, externally calibrated Phase 1 submission."""
import argparse
from pathlib import Path
import hashlib
from sequence_v2 import write
from sequence_v15_core import decode_v15_core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission/Mes_Que_Data_XI.csv"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    write(args.out, decode_v15_core())
    digest = sha256(args.out)
    print("wrote", args.out)
    print("sha256", digest)

if __name__ == "__main__":
    main()
