import sys, os

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print ('Usage: python %s <input>'%sys.argv[0])
        sys.exit(1)
    with open(sys.argv[1]) as fin:
        min_ppl = float('inf')
        min_epoch = -1
        for line in fin:
            if line.startswith('Epoch: '):
                epoch = int(line[len('Epoch: '):].strip())
            if line.startswith('Val Result: PPL:'):
                val_ppl = float(line[len('Val Result: PPL:'):line.find(',')].strip())
                if val_ppl < min_ppl:
                    min_ppl = val_ppl
                    min_epoch = epoch
        print (min_ppl, min_epoch)
       
