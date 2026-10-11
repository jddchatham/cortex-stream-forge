#!/usr/bin/env python3
import argparse, sys
from forge.memory_gate import result_json, verify_hard_memory_gate
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--memory-max", type=int, default=64*1024*1024)
    p.add_argument("--memory-swap-max", type=int, default=0)
    p.add_argument("--timeout", type=float, default=20)
    a = p.parse_args(); r = verify_hard_memory_gate(a.memory_max, a.memory_swap_max, a.timeout); print(result_json(r)); return 0 if r.verified else 1
if __name__ == "__main__": sys.exit(main())
