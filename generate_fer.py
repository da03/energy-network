import sys, os
import math

with open(sys.argv[1]) as fsrc:
    with open(sys.argv[2]) as ftrg:
        with open('uniform.fer', 'w') as ffer:
            for src, trg in zip(fsrc, ftrg):
                l1 = len(src.strip().split())
                l2 = len(trg.strip().split())
                ratio = float(l2) / l1
                sofar = 0
                out = []
                for i in range(l1):
                    new  = min(l2, ratio*(i+1))
                    out.append(str(int(math.ceil(new)) - sofar))
                    sofar = int(math.ceil(new))
                ffer.write(' '.join(out)+'\n')
