import argparse
import sys
import math

import visdom
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
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
                    choices=["soft", "var", "hard"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--split", default="train",
                    choices=["train", "val"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--env", required=True,
                    help="Training model. Default to be vanilla transformer with soft attention.")
#TODO: 
parser.add_argument("--share_decoder_embeddings", default=1,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--share_word_embeddings", default=0,
                    choices=[1, 0],  help="Share src trg embeddings or not.")
parser.add_argument("--max_src_len", type=int, default=15, help="Maximum Src Length")
parser.add_argument("--max_trg_len", type=int, default=15, help="Maximum Trg Length")
parser.add_argument("--min_src_len", type=int, default=13, help="Maximum Src Length")
parser.add_argument("--min_trg_len", type=int, default=13, help="Maximum Trg Length")
parser.add_argument("--batch_size", type=int, default=1, help="Number of tokens per minibatch")
parser.add_argument("--epochs", type=int, default=15, help="Number of Epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate")
parser.add_argument("--temperature", type=float, default=0.1, help="Gumbel Softmax temperature")
parser.add_argument("--train_from", default="", help="Model Path")
parser.add_argument("--dependent_posterior", default=0, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
opts = parser.parse_args()


def main(opts):
    if opts.train_from != '':
        print ('Loading Model from %s'%opts.train_from)
        checkpoint = torch.load(opts.train_from)
        for k in checkpoint['opts'].__dict__:
            if k not in ['train_from', 'batch_size', 'epochs', 'env', 'max_src_len', 'min_src_len', 'max_trg_len', 'min_trg_len']:
                opts.__dict__[k] = checkpoint['opts'].__dict__[k]
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
        return opts.min_src_len <= len(example.src) <= opts.max_src_len \
            and opts.min_trg_len <= len(example.trg) <= opts.max_trg_len 
    train, val = torchtext.datasets.TranslationDataset.splits(
            path=opts.dataset, train='train/train.tags.en-de.bpe',
            validation='dev/valid.en-de.bpe', test=None,
            exts=exts, filter_pred=filter_pred,
            fields=[('src', SRC), ('trg', TRG)])

    BATCH_SIZE = 1
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)), shuffle=True,
                            train=True)
    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=False)
    TRG.vocab = checkpoint['trg_vocab']
    SRC.vocab = TRG.vocab

    # Build Model
    src_vocab_size = len(SRC.vocab.itos)
    trg_vocab_size = len(TRG.vocab.itos)
    print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    print ('Building Model')
    model = make_model(opts.mode, src_vocab_size, trg_vocab_size, n_enc=5, n_dec=5,
                   d_model=278, d_ff=507, h=2, dropout=0.1, share_decoder_embeddings=opts.share_decoder_embeddings,
                   share_word_embeddings=opts.share_word_embeddings, dependent_posterior=opts.dependent_posterior,
                   sharelstm=opts.sharelstm, residual_var=opts.residual_var)
    model.trg_pad = TRG.vocab.stoi["<blank>"]
    print (model)
    if opts.train_from != '':
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
    if opts.split == 'train':
        it = train_iter
    elif opts.split == 'val':
        it = val_iter
    print ('Validation')
    model.eval()
    #val_loss_all = 0
    #val_tokens = 0
    #val_total_correct = 0
    #val_total_nonpadding = 0
    with torch.no_grad():
        for i, batch in enumerate(it):
            num_steps += 1
            src = batch.src[0].transpose(0, 1) # batch, len
            trg = batch.trg.transpose(0, 1) # batch, len
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
            trg_y = trg[:, 1:]
            ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
            decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions = model(src, trg, src_mask, trg_mask, 0)
            samples = model.samples
            outputs = model.generator(decoder_output)[0]
            log_probs = outputs.gather(1, trg_y.view(-1, 1)).view(-1)
            probs = log_probs.exp()
            vis = visdom.Visdom(env='%s_%s'%(opts.split, opts.env))
            if opts.mode == 'var' or opts.mode == 'lstmvar':
                assert samples is not None
                attentions = posterior_attentions
                attentions0 = prior_attentions
                if opts.mode == 'hard':
                    attentions = prior_attentions
                attentions = [attention[0] for attention in attentions]
                attentions0 = [attention[0] for attention in attentions0]
                samples = [sample[0] for sample in samples]
                num_layers = len(attentions)
                num_heads = attentions[0].size(0)
                subplot_titles = []
                for l in range(num_layers):
                    for h in range(num_heads):
                        title = 'layer: %d, head: %d'%(l, h)
                        subplot_titles.append(title + ' priors')
                        subplot_titles.append(title + ' posteriors-priors')
                        subplot_titles.append(title + ' posteriors')
                        subplot_titles.append(title + ' sample')
                fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=4, subplot_titles=subplot_titles)
                fig['layout'].update(width=600*4, height=600*num_layers*num_heads, autosize=False, showlegend=False)
                for l in range(num_layers):
                    for h in range(num_heads):
                        attention = attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        attention0 = attentions0[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        sample = samples[l][h].transpose(0, 1).cpu().data.numpy()
                        row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                        print (row_names)
                        col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                        title = 'layer: %d, head: %d'%(l, h)
                        f = go.Heatmap(z=attention0, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' priors', showscale=False, showlegend=False)
                        fig.append_trace(f, l*num_heads+h+1, 1)
                        f = go.Heatmap(z=(attention-attention0), x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' priors', showscale=False, showlegend=False)
                        fig.append_trace(f, l*num_heads+h+1, 2)
                        f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' posteriors', showscale=False, showlegend=False)
                        #f.legendgroup(title + ' probs')
                        fig.append_trace(f, l*num_heads+h+1, 3)
                        f = go.Heatmap(z=sample, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' sample', showscale=False, showlegend=False)
                        #f.legendgroup(title + ' sample')
                        fig.append_trace(f, l*num_heads+h+1, 4)
                fig['layout'].update(width=600*2, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)


                #fig['layout'] = layout
                vis.plotlyplot(fig)
            elif opts.mode == 'hard':
                assert samples is not None
                attentions = posterior_attentions
                if opts.mode == 'hard':
                    attentions = prior_attentions
                attentions = [attention[0] for attention in attentions]
                samples = [sample[0] for sample in samples]
                num_layers = len(attentions)
                num_heads = attentions[0].size(0)
                subplot_titles = []
                for l in range(num_layers):
                    for h in range(num_heads):
                        title = 'layer: %d, head: %d'%(l, h)
                        subplot_titles.append(title + ' probs')
                        subplot_titles.append(title + ' sample')
                fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=2, subplot_titles=subplot_titles)
                fig['layout'].update(width=600*2, height=600*num_layers*num_heads, autosize=False, showlegend=False)
                for l in range(num_layers):
                    for h in range(num_heads):
                        attention = attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        sample = samples[l][h].transpose(0, 1).cpu().data.numpy()
                        row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                        print (row_names)
                        col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                        title = 'layer: %d, head: %d'%(l, h)
                        f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                        #f.legendgroup(title + ' probs')
                        fig.append_trace(f, l*num_heads+h+1, 1)
                        f = go.Heatmap(z=sample, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' sample', showscale=False, showlegend=False)
                        #f.legendgroup(title + ' sample')
                        fig.append_trace(f, l*num_heads+h+1, 2)
                fig['layout'].update(width=600*2, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)


                #fig['layout'] = layout
                vis.plotlyplot(fig)
            else:
                attentions = prior_attentions
                attentions = [attention[0] for attention in attentions]
                num_layers = len(attentions)
                num_heads = attentions[0].size(0)
                subplot_titles = []
                for l in range(num_layers):
                    for h in range(num_heads):
                        title = 'layer: %d, head: %d'%(l, h)
                        subplot_titles.append(title + ' probs')
                fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=1, subplot_titles=subplot_titles)
                fig['layout'].update(width=600, height=600*num_layers*num_heads, autosize=False, showlegend=False)
                for l in range(num_layers):
                    for h in range(num_heads):
                        attention = attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                        print (row_names)
                        col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                        title = 'layer: %d, head: %d'%(l, h)
                        f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                        fig.append_trace(f, l*num_heads+h+1, 1)
                fig['layout'].update(width=600, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)


                vis.plotlyplot(fig)
            break

if __name__ == '__main__':
    main(opts)
