import sys, os

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('Usage: python %s <input> <output>'%sys.argv[0])
        sys.exit(1)

    with open(sys.argv[1]) as fin:
        with open(sys.argv[2], 'w') as fout:
            words = []
            for line in fin:
                line = line.strip()
                if len(line) == 0:
                    fout.write(' '.join(words)+'\n')
                    words = []
                    continue
                items = line.split('\t')
                words.append(items[1])
