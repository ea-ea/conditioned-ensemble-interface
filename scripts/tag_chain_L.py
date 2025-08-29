#!/usr/bin/env python3
import sys
from pathlib import Path

def tag_chain_L(in_pdb: Path, out_pdb: Path):
    with in_pdb.open() as f, out_pdb.open("w") as g:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                L = line.rstrip("\n")
                if len(L) < 22:
                    L = L + " " * (22 - len(L))
                L = L[:21] + "L" + L[22:]
                g.write(L + "\n")
            else:
                g.write(line)

def main():
    if len(sys.argv) != 3:
        print("usage: tag_chain_L.py <in.pdb> <out.pdb>")
        sys.exit(1)
    tag_chain_L(Path(sys.argv[1]), Path(sys.argv[2]))
    print("[ok] wrote", sys.argv[2])

if __name__ == "__main__":
    main()
