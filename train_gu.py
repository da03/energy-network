import argparse
import sys
import copy
import math
import time 
import numpy as np
import torch
import torchtext
import torch.optim as optim

from data import *
from gu_model import Transformer, FastTransformer
from loss import *
from optimizer import *
import utils

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
UNK = 0
BOS_WORD = '<s>'
EOS_WORD = '</s>'

parser = argparse.ArgumentParser(
    description='train.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", required=True, choices=["Transformer", "FastTransformer", "OldFast"], help="Transformer or FastTransformer or OldFast")
parser.add_argument("--dataset", required=True, help="Path to the dataset")
parser.add_argument("--direction", default="ende",
                    choices=["ende", "deen"],  help="Direction of translation")
parser.add_argument("--max_src_len", type=int, default=150, help="Maximum Src Length")
parser.add_argument("--max_trg_len", type=int, default=150, help="Maximum Trg Length")
parser.add_argument("--epochs", type=int, default=15, help="Number of Epochs")
parser.add_argument("--steps", type=int, default=5, help="Number of Steps")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning Rate")
parser.add_argument("--train_from", default="", help="Model Path")
parser.add_argument("--teacher_model", default="", help="Model Path")
opts = parser.parse_args()

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.numel() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.data.detach())
def main(opts):
    if opts.direction == 'ende':
        exts = ['.en', '.de', '.uniformfer']
    elif opts.direction == 'deen':
        exts = ['.de', '.en', '.uniformfer']
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

    #FER = torchtext.data.Field(sequential=True, preprocessing=torchtext.data.Pipeline(lambda tok: int(tok)), use_vocab=False, pad_token=0, batch_first=True)
    FER = torchtext.data.Field(sequential=True, preprocessing=torchtext.data.Pipeline(lambda tok: 1), use_vocab=False, pad_token=0, batch_first=True)
    train, val = utils.ParallelDataset.splits(
            path=opts.dataset, train='train/train.tags.en-de.bpe',
            validation='dev/valid.en-de.bpe', test=None,
            exts=exts, filter_pred=filter_pred,
            fields=[('src', SRC), ('trg', TRG), ('fer', FER)])

    def batch_size_fn(new, count, sofar):
        """
        In token batching scheme, the number of sequences is limited
        such that the total number of src/trg tokens (including padding)
        in a batch <= batch_size
        """
        return sofar + len(new.src)
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

    BATCH_SIZE = 2048
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    SRC.build_vocab(train)
    TRG.build_vocab(train)

    # Build Model
    src_vocab_size = len(SRC.vocab.itos)
    trg_vocab_size = len(TRG.vocab.itos)
    print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    #               d_model=512, d_ff=2048, h=8, dropout=0.1)
    #model = make_model(src_vocab_size, trg_vocab_size, n_enc=6, n_dec=6, n_inf=1, steps=opts.steps,
    #               d_model=278, d_ff=507, h=2, dropout=0.079)
    args = opts
    args.d_model = 278
    args.d_hidden = 507
    args.n_heads = 2
    args.drop_ratio = 0.079
    args.use_wo = True
    args.length_ratio = 2
    args.input_orderless = False
    args.share_embeddings = False
    args.n_layers = 5
    args.positional_attention = True
    args.diag = True
    args.windows = None
    args.fertility = True # only affects prepare_initial
    args.hard_inputs = False
    args.old = False
    if args.model == "Transformer":
        model = Transformer(SRC, TRG, args)
    else:
        model = FastTransformer(SRC, TRG, args)
    if opts.teacher_model != '':
        teacher_model = Transformer(SRC, TRG, args)
        print ('Loading Teacher Model from %s'%opts.teacher_model)
        checkpoint = torch.load(opts.teacher_model)
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.cuda()
        model.encoder = copy.deepcopy(teacher_model.encoder)
    else:
        teacher_model = None 


    print ('Building Model')
    #model = make_model(src_vocab_size, trg_vocab_size, n_enc=6, n_dec=6, n_inf=1, steps=opts.steps,
    if opts.train_from != '':
        print ('Loading Model from %s'%opts.train_from)
        checkpoint = torch.load(opts.train_from)
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        optimizer = checkpoint['optimizer']
    else:
        model.cuda()
        optimizer = get_std_opt(model, 746, args.d_model, args.model)
    devices = [0]
    #devices = [0, 1, 2, 3]
    model_par = nn.DataParallel(model, device_ids=devices)

    weight = torch.ones(trg_vocab_size)
    weight[TRG.vocab.stoi["<blank>"]] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    #criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=TRG.vocab.stoi["<blank>"], smoothing=0.1)
    criterion.cuda()
                                   
    total_loss = 0.
    total_xent = 0.
    total_kl = 0.
    total_tokens = 0
    total_nonpadding = 0
    total_correct = 0
    loss_all = 0.
    loss_xent = 0.
    loss_kl = 0.
    tokens = 0
    num_steps = 0
    #generator = Generator(d_model, trg_vocab))
    loss_compute = MultiGPULossCompute(TRG, model.decoder.out, criterion, devices=devices, optimizer=optimizer)
    loss_compute.alpha = 0
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
    for epoch in range(opts.epochs):
        print ('')
        print ('Epoch: %d' %epoch)
        print ('Training')
        model.train()
        start = time.time()
        for i, batch in enumerate(train_iter):
            #break
            num_steps += 1
            loss_compute.alpha = min(1.0, float(num_steps) / 10000)
            src = batch.src[0].transpose(0, 1) # batch, len
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_raw = batch.trg.transpose(0, 1) # batch, len
            #trg_padded = trg_raw.new_full((trg_raw.size()[0], opts.max_trg_len), TRG.vocab.stoi["<blank>"])
            #trg_padded = trg_raw.new_full((trg_raw.size()[0], max(src.size()[1], trg_raw.size()[1])), TRG.vocab.stoi["<blank>"])
            trg_padded = trg_raw.new_full((trg_raw.size()[0], max(src.size(1), trg_raw.size()[1])), TRG.vocab.stoi["<blank>"])
            trg = trg_padded.to(trg_raw)
            trg[:, :trg_raw.size()[1]] = trg_raw
            trg_y = trg[:, 1:]
            "Create a mask to hide padding and future words."
            trg_mask = (trg != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
            #import pdb; pdb.set_trace()
            fer = batch.fer
            batch_fer = fer.new_full((fer.size(0), max(src.size(1), trg.size(1))), 0).long()
            batch_fer[:, :fer.size(1)] = fer
            ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()



            sources = src.new_full((src.size(0), max(src.size(1), trg.size(1))), SRC.vocab.stoi["<blank>"])
            sources[:, :src.size(1)] = src
            source_masks = model.prepare_masks(sources)
            encoding = model.encoding(sources, source_masks)
            if args.model == "Transformer":
                inputs = trg
                input_masks = model.prepare_masks(inputs, True)
            elif args.model == "OldFast":
                inputs = encoding[0]
                input_masks = model.prepare_masks(inputs)
            else:
                inputs = encoding[0]
                input_masks = model.prepare_masks(inputs)
                inputs, input_reorder, input_masks, fertility_cost = model.prepare_initial(encoding, sources, source_masks, input_masks, batch_fer)
            #decoder_output, mus, sigmas, logsigmas, decoder_mus, decoder_sigmas, decoder_logsigmas = model_par(src, trg, src_mask, opts.steps, opts.max_trg_len, trg_mask)
            #import pdb; pdb.set_trace()
            decoder_output = model(encoding, source_masks, inputs, input_masks)
            #import pdb; pdb.set_trace()
            if args.model == 'Transformer' or args.model == 'OldFast':
                l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output[:, :-1, :], trg_y, ntokens, None, None, None, None)
            else:
                if decoder_output.size(1) > trg_y.size(1):
                    decoder_output = decoder_output[:, :trg_y.size(1)]
                l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output, trg_y[:, :decoder_output.size(1)], ntokens, None, None, None, None)
            total_loss += l
            total_xent += l_xent
            total_kl += l_kl
            total_correct += l_correct
            #import pdb; pdb.set_trace()
            total_nonpadding += l_nonpadding
            loss_all += l
            loss_xent += l_xent
            loss_kl += l_kl
            total_tokens += ntokens
            tokens += ntokens
            if i % 50 == 1:
                elapsed = max(0.1, time.time() - start)
                #print ('sigma')
                #print (sigmas[0].mean())
                #print ('decoder sigma')
                #print (decoder_sigmas[0].mean())
                #print ('sigma')
                #print (sigmas[1].mean())
                #print ('decoder sigma')
                #print (decoder_sigmas[1].mean())
                print("Epoch Step: %d PPL: %f, Acc: %f, exp xent: %f, xent: %f, kl: %f. Tokens per Sec: %f" %
                (i, math.exp(min(100, loss_all / tokens)), float(total_correct)/total_nonpadding,  math.exp(min(100, loss_xent / tokens)), loss_xent/tokens, loss_kl/tokens, tokens / elapsed))
                sys.stdout.flush()
                start = time.time()
                tokens = 0
                loss_all = 0.
                loss_xent = 0.
                loss_kl = 0.
                total_correct = 0
                total_nonpadding = 0

        print ('Validation')
        model.eval()
        val_loss_all = 0
        val_tokens = 0
        val_total_correct = 0
        val_total_nonpadding = 0
        with torch.no_grad():
            for i, batch in enumerate(val_iter):
                num_steps += 1
                src = batch.src[0].transpose(0, 1) # batch, len
                src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
                trg_raw = batch.trg.transpose(0, 1) # batch, len
                trg_padded = trg_raw.new_full((trg_raw.size()[0], max(src.size(1), trg_raw.size()[1])), TRG.vocab.stoi["<blank>"])
                trg = trg_padded.to(trg_raw)
                trg[:, :trg_raw.size()[1]] = trg_raw
                trg_y = trg[:, 1:]
                "Create a mask to hide padding and future words."
                trg_mask = (trg != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
                trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
                #import pdb; pdb.set_trace()
                fer = batch.fer
                batch_fer = fer.new_full((fer.size(0), max(src.size(1), trg.size(1))), 0).long()
                batch_fer[:, :fer.size(1)] = fer
                ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()



                sources = src.new_full((src.size(0), max(src.size(1), trg.size(1))), SRC.vocab.stoi["<blank>"])
                sources[:, :src.size(1)] = src
                source_masks = model.prepare_masks(sources)
                encoding = model.encoding(sources, source_masks)
                if args.model == "Transformer":
                    inputs = trg
                    input_masks = model.prepare_masks(inputs, True)
                elif args.model == "OldFast":
                    inputs = encoding[0]
                    input_masks = model.prepare_masks(inputs)
                else:
                    inputs = encoding[0]
                    inputs, input_reorder, input_masks, fertility_cost = model.prepare_initial(encoding, sources, source_masks, input_masks, batch_fer)
                decoder_output = model(encoding, source_masks, inputs, input_masks)
                optimizer = loss_compute.optimizer
                loss_compute.optimizer = None
                if args.model == 'Transformer' or args.model == 'OldFast':
                    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output[:, :-1, :], trg_y, ntokens, None, None, None, None)
                else:
                    if decoder_output.size(1) > trg_y.size(1):
                        decoder_output = decoder_output[:, :trg_y.size(1)]
                    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output, trg_y[:, :decoder_output.size(1)], ntokens, None, None, None, None)

                loss_compute.optimizer = optimizer
                val_loss_all += l
                val_tokens += ntokens
                val_total_correct += l_correct
                val_total_nonpadding += l_nonpadding
            print("Val Result: PPL: %f, Acc: %f." %
                (math.exp(min(100, val_loss_all / val_tokens)), float(val_total_correct)/val_total_nonpadding))
        #        src = batch.src[0].transpose(0, 1) # batch, len
        #        print (src.size())
        #        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        #        trg_raw = batch.trg.transpose(0, 1) # batch, len
        #        trg_padded = trg_raw.new_full((trg_raw.size()[0], opts.max_trg_len), TRG.vocab.stoi["<blank>"])
        #        trg = trg_padded.to(trg_raw)
        #        trg[:, :trg_raw.size()[1]] = trg_raw
        #        trg_y = trg[:, 1:]
        #        ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.sum()
        #        decoder_mu = model(src, trg, src_mask, opts.steps, opts.max_trg_len)
        #        decoder_log_probs = model.generator(decoder_mu)
        #        _, token_ids = decoder_log_probs.max(2)
        #        for j in range(token_ids.size()[0]):
        #            src_tokens = []
        #            for k in range(src.size()[1]):
        #                token = SRC.vocab.itos[src[j][k]]
        #                if token == PAD_WORD:
        #                    break
        #                src_tokens.append(token)
        #            print ('src: ' + ' '.join(src_tokens))
        #            trg_tokens = []
        #            for k in range(trg_raw.size()[1]):
        #                token = TRG.vocab.itos[trg_raw[j][k]]
        #                if token == PAD_WORD:
        #                    break
        #                trg_tokens.append(token)
        #            print ('ground truth trg: ' + ' '.join(trg_tokens))
        #            tokens = []
        #            for k in range(token_ids.size()[1]):
        #                token = TRG.vocab.itos[token_ids[j][k]]
        #                if token == EOS_WORD:
        #                    break
        #                tokens.append(token)
        #            print ('predicted trg: ' + ' '.join(tokens))
        torch.save({'model': model.state_dict(), 'opts': opts, 'optimizer': optimizer}, '2048checkpoint.e%d.pt'%epoch)
        model.train()

if __name__ == '__main__':
    main(opts)
