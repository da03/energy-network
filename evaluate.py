import sys
import os
import codecs
from gu_utils import *

tokenizer = lambda x: x.replace('@@ ', '').split()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ('Usage: python %s <predicted> <groundtruth>'%sys.argv[0])
        sys.exit(1)

    with codecs.open(sys.argv[1], 'r', 'utf-8') as fin1:
        with codecs.open(sys.argv[2], 'r', 'utf-8') as fin2:
            predicted = fin1.readlines()
            predicted = [item.strip() for item in predicted]
            groundtruth = fin2.readlines()
            groundtruth = [item.strip() for item in groundtruth]

            corpus_bleu = computeBLEU(predicted, groundtruth, corpus=True, tokenizer=tokenizer)
            print ('BLEU: %f'%corpus_bleu)
