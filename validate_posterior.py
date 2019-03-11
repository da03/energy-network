import argparse
import sys
import math

import visdom
import torch
import torchtext
import torch.optim as optim
import random

random.seed(1234)

from data import *
from model import *
from loss import *
from optimizer import *

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
parser.add_argument("--mode", default="soft",
                    choices=["soft", "var"],  help="Training model. Default to be vanilla transformer with soft attention.")
#TODO: 
parser.add_argument("--share_decoder_embeddings", default=1,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--share_word_embeddings", default=0,
                    choices=[1, 0],  help="Share src trg embeddings or not.")
parser.add_argument("--max_src_len", type=int, default=150, help="Maximum Src Length")
parser.add_argument("--max_trg_len", type=int, default=150, help="Maximum Trg Length")
parser.add_argument("--batch_size", type=int, default=1, help="Number of tokens per minibatch")
parser.add_argument("--epochs", type=int, default=15, help="Number of Epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate")
parser.add_argument("--temperature", type=float, default=0.1, help="Gumbel Softmax temperature")
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
            path=opts.dataset, train='train/train.tags.en-de.bpe',
            validation='dev/valid.en-de.bpe', test=None,
            exts=exts, filter_pred=filter_pred,
            fields=[('src', SRC), ('trg', TRG)])

    BATCH_SIZE = opts.batch_size
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), shuffle=True,
                            train=True)
    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=False)
    TRG.build_vocab(train.src, train.trg)
    SRC.vocab = TRG.vocab

    # Build Model
    src_vocab_size = len(SRC.vocab.itos)
    trg_vocab_size = len(TRG.vocab.itos)
    print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    print ('Building Model')
    model = make_model(opts.mode, src_vocab_size, trg_vocab_size, n_enc=5, n_dec=5,
                   d_model=278, d_ff=507, h=2, dropout=0.1, share_decoder_embeddings=opts.share_decoder_embeddings,
                   share_word_embeddings=opts.share_word_embeddings)
    print (model)
    if opts.train_from != '':
        print ('Loading Model from %s'%opts.train_from)
        checkpoint = torch.load(opts.train_from)
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
    else:
        optimizer = get_std_opt(model, 746, d_model=model.src_embed[0].d_model)
    devices = [0]
    model.cuda()
    print (opts)

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
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
    #print ('')
    #print ('Epoch: %d' %epoch)
    #print ('Training')
    #model.train()
    #start = time.time()
    #for i, batch in enumerate(train_iter):
    #    num_steps += 1
    #    src = batch.src[0].transpose(0, 1) # batch, len
    #    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    #    trg = batch.trg.transpose(0, 1) # batch, len
    #    trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
    #    trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
    #    trg_y = trg[:, 1:]
    #    ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()

    #    decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions = model_par(src, trg, src_mask, trg_mask, opts.temperature)
    #    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output, trg_y, ntokens, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
    #    total_loss += l
    #    total_xent += l_xent
    #    total_kl += l_kl
    #    total_correct += l_correct
    #    total_nonpadding += l_nonpadding
    #    loss_all += l
    #    loss_xent += l_xent
    #    loss_kl += l_kl
    #    total_tokens += ntokens
    #    tokens += ntokens
    #    if i % 50 == 1:
    #        elapsed = max(0.1, time.time() - start)
    #        print("Epoch Step: %d PPL: %f, Acc: %f, exp xent: %f, xent: %f, kl: %f. Tokens per Sec: %f" %
    #        (i, math.exp(min(100, loss_all / tokens)), float(total_correct)/total_nonpadding,  math.exp(min(100, loss_xent / tokens)), loss_xent/tokens, loss_kl/tokens, tokens / elapsed))
    #        sys.stdout.flush()
    #        start = time.time()
    #        tokens = 0
    #        loss_all = 0.
    #        loss_xent = 0.
    #        loss_kl = 0.
    #        total_correct = 0
    #        total_nonpadding = 0

    print ('Validation')
    model.eval()
    #val_loss_all = 0
    #val_tokens = 0
    #val_total_correct = 0
    #val_total_nonpadding = 0
    with torch.no_grad():
        for i, batch in enumerate(train_iter):
            num_steps += 1
            src = batch.src[0].transpose(0, 1) # batch, len
            trg = batch.trg.transpose(0, 1) # batch, len
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
            trg_y = trg[:, 1:]
            ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
            #decoder_output, prior_attentions, posterior_attentions = model_par(src, trg, src_mask, trg_mask)
            #optimizer = loss_compute.optimizer
            #loss_compute.optimizer = None
            #import pdb; pdb.set_trace()
            decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions = model(src, trg, src_mask, trg_mask, opts.temperature)
            prior_attentions = posterior_attentions
            prior_attentions = [prior_attention[0] for prior_attention in prior_attentions]
            num_layers = len(prior_attentions)
            num_heads = prior_attentions[0].size(0)
            vis = visdom.Visdom()
            for l in range(num_layers):
                for h in range(num_heads):
                    attention = prior_attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                    row_names = [SRC.vocab.itos[i] for i in src.view(-1).cpu().data.numpy()]
                    col_names = [TRG.vocab.itos[i] for i in trg_y.view(-1).cpu().data.numpy()]
                    title = 'layer: %d, head: %d'%(l, h)
                    vis.heatmap(
                                 X=attention,
                                 opts=dict(
                                     rownames=row_names,
                                     columnnames=col_names,
                                     colormap="Hot",
                                     title=title,
                                     width=750,
                                     height=750,
                                     marginleft=150,
                                     marginright=150,
                                     margintop=150,
                                     marginbottom=150
                                 ),
                                 win=title
                             )
            
            break
            #l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output, trg_y, ntokens, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
            #decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions = model_par(src, trg, src_mask, trg_mask, 0)
            #word_probs = model.generator(decoder_output)

            #l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output, trg_y, ntokens, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
            #l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute(decoder_output, trg_y, ntokens, mus, sigmas, decoder_mus, decoder_sigmas)
            #loss_compute.optimizer = optimizer
            #val_loss_all += l
            #val_tokens += ntokens
            #val_total_correct += l_correct
            #val_total_nonpadding += l_nonpadding
        #print("Val Result: PPL: %f, Acc: %f." %
        #    (math.exp(min(100, val_loss_all / val_tokens)), float(val_total_correct)/val_total_nonpadding))
        #torch.save({'model': model.state_dict(), 'src_vocab': SRC.vocab, 'trg_vocab': TRG.vocab, 'opts': opts, 'optimizer': optimizer}, '%s.e%d.pt'%(opts.save_to, epoch))
    model.train()

if __name__ == '__main__':
    main(opts)
