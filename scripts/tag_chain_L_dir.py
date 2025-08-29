#!/usr/bin/env python3
import sys, glob
from pathlib import Path

def tag_L(in_pdb: Path, out_pdb: Path):
    with in_pdb.open() as f, out_pdb.open("w") as g:
        for line in f:
            if line.startswith(("ATOM","HETATM")):
                L = line.rstrip("\n")
                if len(L) < 22:
                    L = L + " " * (22 - len(L))
                L = L[:21] + "L" + L[22:]
                g.write(L + "\n")
            else:
                g.write(line)

def main():
    if len(sys.argv) < 2:
        print("usage: tag_chain_L_dir.py <glob like runs/3ptb_smina/pose_*.pdb>")
        sys.exit(1)
    count = 0
    for pat in sys.argv[1:]:
        for p in sorted(glob.glob(pat)):
            if p.endswith("_L.pdb"):
                continue
            inp = Path(p)
            out = inp.with_name(inp.stem + "_L.pdb")
            tag_L(inp, out)
            count += 1
    print(f"[ok] tagged {count} files with chain L")

if __name__ == "__main__":
    main()
