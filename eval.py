import argparse
import sys
import math

import torch
import torchtext
import torch.optim as optim

from model import *
from loss import *
from optimizer import *
from data import *

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, help="Path to the dataset")
parser.add_argument("--direction", default="ende",
                    choices=["ende", "deen"],  help="Direction of translation")
parser.add_argument("--max_src_len", type=int, default=150, help="Maximum Src Length")
parser.add_argument("--max_trg_len", type=int, default=150, help="Maximum Trg Length")
parser.add_argument("--epochs", type=int, default=15, help="Number of Epochs")
parser.add_argument("--steps", type=int, default=5, help="Number of Steps")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
parser.add_argument("--train_from", default="", help="Model Path")
opts = parser.parse_args()

def main(opts):
    if opts.direction == 'ende':
        exts = ['.en', '.de']
    elif opts.direction == 'deen':
        exts = ['.de', '.en']
    else:
        raise NotImplementedError
    SRC = torchtext.data.Field(
            pad_token=PAD_WORD,
            include_lengths=True)
    TRG = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)
    def filter_pred(example):
        return 0 < len(example.src) <= opts.max_src_len \
            and 0 < len(example.trg) <= opts.max_trg_len 
    train, val = torchtext.datasets.TranslationDataset.splits(
            path=opts.dataset, train='train/train.tags.en-de.bpe.head',
            validation='dev/valid.en-de.bpe', test=None,
            exts=exts, filter_pred=filter_pred,
            fields=[('src', SRC), ('trg', TRG)])

    def batch_size_fn(new, count, sofar):
        """
        In token batching scheme, the number of sequences is limited
        such that the total number of src/trg tokens (including padding)
        in a batch <= batch_size
        """
        # Maintains the longest src and trg length in the current batch
        global max_src_in_batch, max_trg_in_batch
        # Reset current longest length at a new batch (count=1)
        if count == 1:
            max_src_in_batch = 0
            max_trg_in_batch = 0
        # Src: <bos> w1 ... wN <eos>
        max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
        # Tgt: w1 ... wN <eos>
        max_trg_in_batch = max(max_trg_in_batch, len(new.trg) + 1)
        src_elements = count * max_src_in_batch
        trg_elements = count * max_trg_in_batch
        return max(src_elements, trg_elements)
    BATCH_SIZE = 12000
    #train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
    #                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                        batch_size_fn=batch_size_fn, train=False)
    #val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
    #                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
    #                        batch_size_fn=batch_size_fn, train=False)


    train_iter, val_iter = torchtext.data.BucketIterator.splits(
            (train, val), batch_size=BATCH_SIZE, batch_size_fn=batch_size_fn,
            sort_key=lambda x: len(x.src), device=torch.device('cuda:0'), repeat=False, sort_within_batch=False, sort=False, shuffle=False)
    train, val = torchtext.datasets.TranslationDataset.splits(
            path=opts.dataset, train='train/train.tags.en-de.bpe',
            validation='dev/valid.en-de.bpe', test=None,
            exts=exts, filter_pred=filter_pred,
            fields=[('src', SRC), ('trg', TRG)])
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    # Build Model
    src_vocab_size = len(SRC.vocab.itos)
    trg_vocab_size = len(TRG.vocab.itos)
    print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    if opts.train_from != '':
        print ('Loading Model from %s'%opts.train_from)
        checkpoint = torch.load(opts.train_from)
        model = checkpoint['model']
        model.cuda()
        optimizer = checkpoint['optimizer']
    else:
        assert False
        print ('Building Model')
        model = make_model(src_vocab_size, trg_vocab_size, n_enc=6, n_dec=6, n_inf=1, steps=opts.steps,
                       d_model=512, d_ff=2048, h=8, dropout=0.1)
        model.cuda()
        optimizer = get_std_opt(model)
    #devices = [0]
    devices = [0, 1, 2, 3]
    model_par = nn.DataParallel(model, device_ids=devices)

    weight = torch.ones(trg_vocab_size)
    weight[TRG.vocab.stoi["<blank>"]] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    criterion.cuda()
                                   
    total_loss = 0.
    total_xent = 0.
    total_kl = 0.
    total_tokens = 0
    total_nonpadding = 0
    total_correct = 0
    loss = 0.
    loss_xent = 0.
    loss_kl = 0.
    tokens = 0
    num_steps = 0
    loss_compute = MultiGPULossCompute(TRG, model.generator, criterion, devices=devices, optimizer=optimizer)
    loss_compute.alpha = 0
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
    print ('')
    model.eval()
    start = time.time()
    tokens = 0
    print ('Validation')
    model.eval()
    with torch.no_grad():
        with open('pred.train.txt', 'w') as fout:
            for i, batch in enumerate(val_iter):
                src = batch.src[0].transpose(0, 1) # batch, len
                src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
                trg_raw = batch.trg.transpose(0, 1) # batch, len
                #trg_padded = trg_raw.new_full((trg_raw.size()[0], max(src.size()[1], trg_raw.size()[1])), TRG.vocab.stoi["<blank>"])
                trg_padded = trg_raw.new_full((trg_raw.size()[0], trg_raw.size()[1]), TRG.vocab.stoi["<blank>"])
                trg = trg_padded.to(trg_raw)
                trg[:, :trg_raw.size()[1]] = trg_raw
                trg_y = trg[:, 1:]
                ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.sum()
                trg_mask = (trg != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
                trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
                decoder_output, mus, sigmas, logsigmas, decoder_mus, decoder_sigmas, decoder_logsigmas = model_par(src, trg.fill_(0), src_mask, opts.steps, opts.max_trg_len, trg_mask=trg_mask)
                decoder_log_probs = model.generator(decoder_output)
                _, token_ids = decoder_log_probs.max(2)
                for j in range(token_ids.size()[0]):
                    src_tokens = []
                    for k1 in range(src.size()[1]):
                        token = SRC.vocab.itos[src[j][k1]]
                        if token == PAD_WORD:
                            break
                        src_tokens.append(token)
                    print ('src: ' + ' '.join(src_tokens))
                    trg_tokens = []
                    for k2 in range(trg_raw.size()[1]):
                        token = TRG.vocab.itos[trg_raw[j][k2]]
                        if token == PAD_WORD:
                            break
                        trg_tokens.append(token)
                    print ('ground truth trg: ' + ' '.join(trg_tokens))
                    tokens = []
                    for k3 in range(token_ids.size()[1]):
                        token = TRG.vocab.itos[token_ids[j][k3]]
                        if k3 > k1 and k3 > k2:
                            break
                        if token == EOS_WORD:
                            break
                        tokens.append(token)
                    print ('predicted trg: ' + ' '.join(tokens))
                    fout.write(' '.join(tokens) + '\n')

if __name__ == '__main__':
    main(opts)
