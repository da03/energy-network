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
parser.add_argument("--mode", required=True,
                    choices=["soft", "var", "hard"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--selfmode", required=True,
                    choices=["soft", "var", "hard", "lstmvar"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--encselfmode", required=True,
                    choices=["soft", "var", "hard", "lstmvar"],  help="Training model. Default to be vanilla transformer with soft attention.")
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
parser.add_argument("--train_from", required=True, help="Model Path")
parser.add_argument("--dependent_posterior", default=0, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--mono", required=True, type=int,  help="Monolingual or not")
opts = parser.parse_args()


def main(opts):
    print ('Loading Model from %s'%opts.train_from)
    checkpoint = torch.load(opts.train_from)
    mono = checkpoint['opts'].mono
    for k in checkpoint['opts'].__dict__:
        if k not in ['train_from', 'batch_size', 'epochs', 'env', 'max_src_len', 'min_src_len', 'max_trg_len', 'min_trg_len']:
            opts.__dict__[k] = checkpoint['opts'].__dict__[k]
    if opts.direction == 'ende':
        exts = ['.en', '.de']
    elif opts.direction == 'deen':
        exts = ['.de', '.en']
    else:
        raise NotImplementedError
    if mono == 0:
        SRC = torchtext.data.Field(
                pad_token=PAD_WORD,
                include_lengths=True)
    TRG = torchtext.data.Field(
            init_token=BOS_WORD, eos_token=EOS_WORD,
            pad_token=PAD_WORD)
    def filter_pred(example):
        return opts.min_src_len <= len(example.src) <= opts.max_src_len \
            and opts.min_trg_len <= len(example.trg) <= opts.max_trg_len 
    if mono == 0:
        train, val = torchtext.datasets.TranslationDataset.splits(
                path=opts.dataset, train='train/train.tags.en-de.bpe',
                validation='dev/valid.en-de.bpe', test=None,
                exts=exts, filter_pred=filter_pred,
                fields=[('src', SRC), ('trg', TRG)])
    else:
        train, val, test = MyLanguageModelingDataset.splits(TRG,
                path=opts.dataset, train='train.tok.txt',
                validation='valid.tok.txt', test='test.tok.txt',
                )

    BATCH_SIZE = 1
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)) if hasattr(x, 'src') else len(x.trg), shuffle=True,
                            train=True)
    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)) if hasattr(x, 'src') else len(x.trg),
                            train=False)
    if opts.mono == 1:
        test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                                repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)) if hasattr(x, 'src') else len(x.trg),
                                train=False)
    TRG.vocab = checkpoint['trg_vocab']
    if mono == 0:
        SRC.vocab = TRG.vocab

    # Build Model
    trg_vocab_size = len(TRG.vocab.itos)
    if mono == 0:
        src_vocab_size = len(SRC.vocab.itos)
        print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    else:
        src_vocab_size = None
        print ('TRG Vocab Size: %d'%(trg_vocab_size))
    print ('Building Model')
    sharelstm = checkpoint['opts'].sharelstm if hasattr(checkpoint['opts'], 'lstm') else 0
    residual_var = checkpoint['opts'].residualvar if hasattr(checkpoint['opts'], 'residualvar') else 0
    model = make_model(opts.mode, src_vocab_size, trg_vocab_size, n_enc=5, n_dec=5,
                   d_model=278, d_ff=507, h=2, dropout=0.1, share_decoder_embeddings=opts.share_decoder_embeddings,
                   share_word_embeddings=opts.share_word_embeddings, dependent_posterior=opts.dependent_posterior,
                   sharelstm=sharelstm, residual_var=residual_var, selfmode=opts.selfmode, encselfmode=opts.encselfmode, mono=mono)
    model.trg_pad = TRG.vocab.stoi["<blank>"]
    model.load_state_dict(checkpoint['model'])
    optimizer = checkpoint['optimizer']
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
            if mono == 0:
                src = batch.src[0].transpose(0, 1) # batch, len
                src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            else:
                src, src_mask = None, None
            trg = batch.trg.transpose(0, 1) # batch, len
            trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
            trg_y = trg[:, 1:]
            ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
            decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_encself_attention_priors, enc_self_attention_priors = model(src, trg, src_mask, trg_mask, 0, selftemperature=0, encselftemperature=0)
            samples = model.samples
            self_attention_samples = model.self_attention_samples
            enc_self_attention_samples = model.enc_self_attention_samples
            outputs = model.generator(decoder_output)[0]
            log_probs = outputs.gather(1, trg_y.view(-1, 1)).view(-1)
            probs = log_probs.exp()
            vis = visdom.Visdom(env='%s_%s'%(opts.split, opts.env))
            tot_cols = 0
            if True:
                if prior_attentions is not None and len(prior_attentions)>0 and prior_attentions[0] is not None:
                    tot_cols += 1
                    prior_attentions = [attention[0] for attention in prior_attentions]
                if posterior_attentions is not None and len(posterior_attentions)>0:
                    tot_cols += 1
                    diff_attentions = [posterior_attention[0]-prior_attention[0] for posterior_attention, prior_attention in zip(posterior_attentions, prior_attentions)]
                if samples is not None and len(samples) > 0:
                    tot_cols += 1
                    samples = [attention[0] for attention in samples]
                if self_attention_priors is not None and len(self_attention_priors)>0:
                    tot_cols += 1
                    self_attention_priors = [attention[0] for attention in self_attention_priors]
                if self_attention_samples is not None and len(self_attention_samples) > 0:
                    tot_cols += 1
                    self_attention_samples = [attention[0] for attention in self_attention_samples]
                if enc_self_attention_priors is not None and len(enc_self_attention_priors)>0 and enc_self_attention_priors[0] is not None:
                    tot_cols += 1
                    enc_self_attention_priors = [attention[0] for attention in enc_self_attention_priors]
                if enc_self_attention_samples is not None and len(enc_self_attention_samples) > 0:
                    tot_cols += 1
                    enc_self_attention_samples = [attention[0] for attention in enc_self_attention_samples]
                
                
                num_layers = len(self_attention_priors)
                num_heads = self_attention_priors[0].size(0)
                subplot_titles = []
                import pdb; pdb.set_trace()
                for l in range(num_layers):
                    for h in range(num_heads):
                        title = 'layer: %d, head: %d'%(l, h)
                        if prior_attentions is not None and len(prior_attentions)>0 and prior_attentions[0] is not None:
                            subplot_titles.append(title + ' prior probs')
                        if posterior_attentions is not None and len(posterior_attentions)>0:
                            subplot_titles.append(title + ' posterior-prior probs')
                        if samples is not None and len(samples) > 0:
                            subplot_titles.append(title + ' sample')
                        if self_attention_priors is not None and len(self_attention_priors)>0:
                            subplot_titles.append(title + ' probs self')
                        if self_attention_samples is not None and len(self_attention_samples) > 0:
                            subplot_titles.append(title + ' sample self')
                        if enc_self_attention_priors is not None and len(enc_self_attention_priors)>0:
                            subplot_titles.append(title + ' probs self enc')
                        if enc_self_attention_samples is not None and len(enc_self_attention_samples) > 0 and enc_self_attention_priors[0] is not None:
                            subplot_titles.append(title + ' sample self enc')
                fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=tot_cols, subplot_titles=subplot_titles)
                fig['layout'].update(width=400*tot_cols, height=600*num_layers*num_heads, autosize=False, showlegend=False)
                for l in range(num_layers):
                    for h in range(num_heads):
                        idx = 0
                        if mono == 0:
                            src_row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                        trg_out_col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                        trg_row_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                        title = 'layer: %d, head: %d'%(l, h)
                        if prior_attentions is not None and len(prior_attentions)>0 and prior_attentions[0] is not None:
                            idx += 1
                            attention = prior_attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=src_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        if posterior_attentions is not None and len(posterior_attentions)>0:
                            idx += 1
                            attention = diff_attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=src_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        if samples is not None and len(samples) > 0:
                            idx += 1
                            attention = samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=src_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        if self_attention_priors is not None and len(self_attention_priors)>0:
                            idx += 1
                            attention = self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            import pdb; pdb.set_trace()
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        if self_attention_samples is not None and len(self_attention_samples) > 0:
                            idx += 1
                            attention = self_attention_samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        if enc_self_attention_priors is not None and len(enc_self_attention_priors)>0 and enc_self_attention_priors[0] is not None:
                            idx += 1
                            attention = enc_self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        if enc_self_attention_samples is not None and len(enc_self_attention_samples) > 0:
                            idx += 1
                            attention = enc_self_attention_samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        #f.legendgroup(title + ' probs')
                fig['layout'].update(width=400*tot_cols, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)


                #fig['layout'] = layout
                vis.plotlyplot(fig)
            elif opts.mode == 'var' or opts.mode == 'lstmvar':
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
                attentions = prior_attentions
                attentions = [attention[0] for attention in attentions]
                self_attention_priors = [attention[0] for attention in self_attention_priors]
                self_attention_samples = model.self_attention_samples
                
                self_attention_samples = [attention[0] for attention in self_attention_samples]
                samples = [sample[0] for sample in samples]
                num_layers = len(attentions)
                num_heads = attentions[0].size(0)
                subplot_titles = []
                for l in range(num_layers):
                    for h in range(num_heads):
                        title = 'layer: %d, head: %d'%(l, h)
                        subplot_titles.append(title + ' probs')
                        subplot_titles.append(title + ' sample')
                        subplot_titles.append(title + ' probs self')
                        subplot_titles.append(title + ' sample self')
                fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=4, subplot_titles=subplot_titles)
                fig['layout'].update(width=400*4, height=600*num_layers*num_heads, autosize=False, showlegend=False)
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
                        row_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                        attention = self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                        fig.append_trace(f, l*num_heads+h+1, 3)
                        attention = self_attention_samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                        fig.append_trace(f, l*num_heads+h+1, 4)
                        #f.legendgroup(title + ' probs')
                fig['layout'].update(width=400*4, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)


                #fig['layout'] = layout
                vis.plotlyplot(fig)
            else: #log_self_attention_priors, self_attention_priorsj
                if opts.selfmode == 'soft':
                    attentions = prior_attentions
                    attentions = [attention[0] for attention in attentions]
                    self_attention_priors = [attention[0] for attention in self_attention_priors]
                    num_layers = len(attentions)
                    num_heads = attentions[0].size(0)
                    subplot_titles = []
                    for l in range(num_layers):
                        for h in range(num_heads):
                            title = 'layer: %d, head: %d'%(l, h)
                            subplot_titles.append(title + ' probs inter')
                            subplot_titles.append(title + ' probs self')
                    fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=2, subplot_titles=subplot_titles)
                    fig['layout'].update(width=600*2, height=600*num_layers*num_heads, autosize=False, showlegend=False)
                    for l in range(num_layers):
                        for h in range(num_heads):
                            attention = attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                            print (row_names)
                            col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            title = 'layer: %d, head: %d'%(l, h)
                            f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, 1)
                            attention = self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            row_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            title = 'layer: %d, head: %d'%(l, h)
                            f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs self', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, 2)
                    fig['layout'].update(width=600*2, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)
                else:
                    attentions = prior_attentions
                    attentions = [attention[0] for attention in attentions]
                    self_attention_samples = model.self_attention_samples
                
                    self_attention_samples = [attention[0] for attention in self_attention_samples]
                    self_attention_priors = [attention[0] for attention in self_attention_priors]
                    num_layers = len(attentions)
                    num_heads = attentions[0].size(0)
                    subplot_titles = []
                    for l in range(num_layers):
                        for h in range(num_heads):
                            title = 'layer: %d, head: %d'%(l, h)
                            subplot_titles.append(title + ' probs inter')
                            subplot_titles.append(title + ' probs self')
                            subplot_titles.append(title + ' samples self')
                    fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=3, subplot_titles=subplot_titles)
                    fig['layout'].update(width=500*3, height=600*num_layers*num_heads, autosize=False, showlegend=False)
                    for l in range(num_layers):
                        for h in range(num_heads):
                            attention = attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                            print (row_names)
                            col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            title = 'layer: %d, head: %d'%(l, h)
                            f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, 1)
                            attention = self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            row_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            title = 'layer: %d, head: %d'%(l, h)
                            f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs self', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, 2)
                            attention = self_attention_samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                            col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            row_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                            title = 'layer: %d, head: %d'%(l, h)
                            f = go.Heatmap(z=attention, x=col_names, y=row_names, colorscale='Hot', legendgroup=title+' probs self', showscale=False, showlegend=False)
                            fig.append_trace(f, l*num_heads+h+1, 3)
                    fig['layout'].update(width=500*3, height=600*num_layers*num_heads, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)


                vis.plotlyplot(fig)
            break

if __name__ == '__main__':
    main(opts)
