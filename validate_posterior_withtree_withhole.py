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

import networkx as nx
import pygraphviz as pgv
import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly.graph_objs import *
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
        vis = visdom.Visdom(env='%s'%(opts.env), port=7098)
        assert vis.check_connection(timeout_seconds=3), \
            'No connection could be formed quickly'

        # text window with Callbacks
        # Properties window
        global default_collapsed
        default_collapsed = 1
        properties = [
            {'type': 'button', 'name': 'Button', 'value': 'Prev'},
            {'type': 'button', 'name': 'Button', 'value': 'Next'},
            {'type': 'select', 'name': 'Bylayer or Collapsed', 'value': default_collapsed, 'values': ['Bylayer', 'Collapsed']},
        ]

        global cur_idx
        cur_idx = 0
        properties_window = vis.properties(properties)
        def properties_callback(event):
            if event['event_type'] == 'PropertyUpdate':
                prop_id = event['propertyId']
                value = event['value']
                global cur_idx
                global default_collapsed
                if prop_id == 0:
                    if cur_idx == 0:
                        return
                    cur_idx -= 1
                elif prop_id == 1:
                    cur_idx += 1
                else:
                    if int(value) == default_collapsed:
                        return
                    properties[prop_id]['value'] = value
                    vis.properties(properties, win=properties_window)
                    default_collapsed = int(value)
                global callback_window1
                global callback_window2
                cur_batch = get_example(cur_idx)
                if cur_batch is None:
                    return
                vis.close(win=callback_window1)
                vis.close(win=callback_window2)
                callback_window1, callback_window2 = draw_batch(cur_batch, collapsed=(default_collapsed==1))
                #properties[prop_id]['value'] = new_value
                #vis.properties(properties, win=properties_window)
                #vis.text("Updated: {} => {}".format(properties[event['propertyId']]['name'], str(new_value)),
                #         win=callback_text_window, append=True)

        vis.register_event_handler(properties_callback, properties_window)
        global global_dict
        global_dict = {}
        it = iter(it)
        def get_example(idx):
            if idx not in global_dict:
                batch = next(it)
                if batch is None:
                    return None
                global_dict[idx] = batch
            return global_dict[idx]

        cur_batch = get_example(cur_idx)

        def draw_batch(batch, collapsed=True):
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
            tot_cols = 0
            self_attention_priors = None
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
                tot_cols += 2
                self_attention_samples = [attention[0] for attention in self_attention_samples]
            if enc_self_attention_priors is not None and len(enc_self_attention_priors)>0 and enc_self_attention_priors[0] is not None:
                tot_cols += 1
                enc_self_attention_priors = [attention[0] for attention in enc_self_attention_priors]
            if enc_self_attention_samples is not None and len(enc_self_attention_samples) > 0:
                tot_cols += 1
                enc_self_attention_samples = [attention[0] for attention in enc_self_attention_samples]
            
            
            num_layers = len(self_attention_samples)
            num_heads = self_attention_samples[0].size(0)
            subplot_titles = []
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
                    if self_attention_samples is not None and len(self_attention_samples) > 0:
                        if h == num_heads-1:
                            subplot_titles.append('layer: %d dependencies'%l)
                        elif h == num_heads-2:
                            subplot_titles.append('layer: %d dependencies with distances'%l)
                        else:
                            subplot_titles.append(title + ' dependencies')
                    if enc_self_attention_priors is not None and len(enc_self_attention_priors)>0:
                        subplot_titles.append(title + ' probs self enc')
                    if enc_self_attention_samples is not None and len(enc_self_attention_samples) > 0 and enc_self_attention_priors[0] is not None:
                        subplot_titles.append(title + ' sample self enc')
            fig = plotly.tools.make_subplots(rows=num_layers*num_heads, cols=tot_cols, subplot_titles=subplot_titles)
            fig['layout'].update(width=700*tot_cols, height=900*num_layers*num_heads, autosize=False, showlegend=False)
            connectivity = {}
            prev_attention_dependencies = None
            prev_attention_dependencies_dist = None
            for l in range(num_layers):
                for h in range(num_heads):
                    idx = 0
                    if mono == 0:
                        src_row_names = [SRC.vocab.itos[i] for i in src.contiguous().view(-1).cpu().data.numpy()]
                    trg_out_col_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                    trg_out_col_names_noppl = [TRG.vocab.itos[i] for i,p in zip(trg_y.contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                    trg_row_names = [TRG.vocab.itos[i]+' %f'%(1/p) for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                    trg_row_names_noppl = [TRG.vocab.itos[i] for i,p in zip(trg[:,:-1].contiguous().view(-1).cpu().data.numpy(), probs.cpu().data.numpy())]
                    title = 'layer: %d, head: %d'%(l, h)
                    if prior_attentions is not None and len(prior_attentions)>0 and prior_attentions[0] is not None:
                        idx += 1
                        attention = prior_attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=src_row_names, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                    if posterior_attentions is not None and len(posterior_attentions)>0:
                        idx += 1
                        attention = diff_attentions[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=src_row_names, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                    if samples is not None and len(samples) > 0:
                        idx += 1
                        attention = samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=src_row_names, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                    if self_attention_priors is not None and len(self_attention_priors)>0:
                        idx += 1
                        attention = self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names_noppl, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                    if self_attention_samples is not None and len(self_attention_samples) > 0:
                        idx += 1
                        attention = self_attention_samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        attention_dependencies_dist = copy.deepcopy(attention)
                        for i in range(len(trg_row_names)):
                            for j in range(len(trg_out_col_names)):
                                input_name = str(i) + ': '+trg_row_names_noppl[i]
                                if l > 0:
                                    input_name = input_name + '_LAYER%d'%(l-1)
                                    residual_name = str(j) + ': '+trg_row_names_noppl[j] + '_LAYER%d'%(l-1)
                                else:
                                    residual_name = str(j) + ': '+trg_row_names_noppl[j]
                                output_name = str(j+1) + ': '+trg_out_col_names_noppl[j]
                                if l != num_layers-1:
                                    output_name = str(j) + ': '+trg_row_names_noppl[j]
                                    #output_name = output_name + '_LAYER%d'%l
                                    output_name = output_name + '_LAYER%d'%l
                                if attention[i][j] == 1:
                                    connectivity.setdefault(output_name, [residual_name])
                                    if residual_name in connectivity:
                                        for name in connectivity[residual_name]:
                                            if name not in connectivity[output_name]:
                                                connectivity[output_name].append(name)
                                    if input_name not in connectivity[output_name]:
                                        connectivity[output_name].append(input_name)
                                    if input_name in connectivity:
                                        for name in connectivity[input_name]:
                                            if name not in connectivity[output_name]:
                                                connectivity[output_name].append(name)
                                    
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names_noppl, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                        idx += 1
                        if h == num_heads-2:
                            for i in range(len(trg_row_names)):
                                for j in range(len(trg_out_col_names)):
                                    if attention_dependencies_dist[i][j] == 1:
                                        attention_dependencies_dist[i][j] = 1
                                    else:
                                        attention_dependencies_dist[i][j] = float('inf')
                                    if i == j:
                                        attention_dependencies_dist[i][j] = 0
                            for h2 in list(range(num_heads-2)) + [h+1]:
                                attention_dependencies2= self_attention_samples[l][h2].transpose(0 ,1).cpu().data.numpy() # trg by src
                                for i in range(len(trg_row_names)):
                                    for j in range(len(trg_out_col_names)):
                                        if attention_dependencies2[i][j] == 1:
                                            attention_dependencies_dist[i][j] = min(1, attention_dependencies_dist[i][j])
          
                            if prev_attention_dependencies_dist is not None:
                                for k in range(len(trg_row_names)):
                                    for i in range(len(trg_row_names)):
                                        for j in range(len(trg_out_col_names)):
                                            if prev_attention_dependencies_dist[k][i] < float('inf') and attention_dependencies_dist[i][j] < float('inf'):
                                                attention_dependencies_dist[k][j] = min(attention_dependencies_dist[k][j], prev_attention_dependencies_dist[k][i]+attention_dependencies_dist[i][j])
                            prev_attention_dependencies_dist = attention_dependencies_dist
                            attention_dependencies_dist2 = copy.deepcopy(attention_dependencies_dist)
                            hovertext = []
                            for i in range(len(trg_row_names)):
                                hovertext_ = []
                                for j in range(len(trg_out_col_names)):
                                    if attention_dependencies_dist2[i][j] == float('inf'):
                                        attention_dependencies_dist2[i][j] = -4.4
                                        hovertext_.append('out: %s<br>in: %s<br>dist: inf'%(trg_out_col_names[j], trg_row_names_noppl[i]))
                                    else:
                                        attention_dependencies_dist2[i][j] = -attention_dependencies_dist2[i][j]
                                        #hovertext_.append('dist: %d'%(-attention_dependencies_dist2[i][j]))
                                        hovertext_.append('out: %s<br>in: %s<br>dist: %d'%(trg_out_col_names[j], trg_row_names_noppl[i], -attention_dependencies_dist2[i][j]))
                                hovertext.append(hovertext_)
                            f = go.Heatmap(z=attention_dependencies_dist2, x=trg_out_col_names, y=trg_row_names_noppl, colorscale='Hot', showscale=False, hovertext=hovertext, hoverinfo="text")
                            fig.append_trace(f, l*num_heads+h+1, idx)
                        elif h == num_heads-1:
                            attention_dependencies = attention
                            for h2 in range(num_heads-1):
                                attention_dependencies2= self_attention_samples[l][h2].transpose(0 ,1).cpu().data.numpy() # trg by src
                                for i in range(len(trg_row_names)):
                                    for j in range(len(trg_out_col_names)):
                                        if i == j:
                                            attention_dependencies[i][j] = 1
                                        if attention_dependencies2[i][j] == 1:
                                            attention_dependencies[i][j] = 1
          
                            if prev_attention_dependencies is not None:
                                for k in range(len(trg_row_names)):
                                    for i in range(len(trg_row_names)):
                                        for j in range(len(trg_out_col_names)):
                                            if prev_attention_dependencies[k][i] == 1 and attention_dependencies[i][j] == 1:
                                                attention_dependencies[k][j] = 1
                            prev_attention_dependencies = attention_dependencies
                                    
                            f = go.Heatmap(z=attention_dependencies, x=trg_out_col_names, y=trg_row_names_noppl, colorscale='Hot', showscale=False)
                            fig.append_trace(f, l*num_heads+h+1, idx)
                            
                    if enc_self_attention_priors is not None and len(enc_self_attention_priors)>0 and enc_self_attention_priors[0] is not None:
                        idx += 1
                        attention = enc_self_attention_priors[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                    if enc_self_attention_samples is not None and len(enc_self_attention_samples) > 0:
                        idx += 1
                        attention = enc_self_attention_samples[l][h].transpose(0 ,1).cpu().data.numpy() # trg by src
                        f = go.Heatmap(z=attention, x=trg_out_col_names, y=trg_row_names, colorscale='Hot', showscale=False)
                        fig.append_trace(f, l*num_heads+h+1, idx)
                    #f.legendgroup(title + ' probs')
            width = 400*tot_cols if tot_cols > 3 else 700*tot_cols
            height = 900*num_layers*num_heads
            fig['layout'].update(width=width, height=height, showlegend=False, autosize=False, title='Mode: %s'%opts.mode)
            H=pgv.AGraph(strict=True, directed=True)
            if collapsed:
                for name in connectivity:
                    for outname in connectivity[name]:
                        if '_LAYER' in outname or '_LAYER' in name:
                            continue
                        if '<s>' in name or '<s>' in outname:
                            continue
                        print ('edge: %s -> %s'%(name, outname))
                        H.add_edge(name, outname)
                H.layout()
                H.draw('/tmp/pyviz_rendered.png', prog='dot')
            else:
                for name in connectivity:
                    for outname in connectivity[name]:
                        if '<s>' in name or '<s>' in outname:
                            continue
                        if '_LAYER' in name:
                            if '_LAYER0' not in name:
                                if '_LAYER' not in outname:
                                    continue
                                idx = name.find('_LAYER')
                                name_l = int(name[(idx+ len('_LAYER')):])
                                idx = outname.find('_LAYER')
                                outname_l = int(outname[(idx+ len('_LAYER')):])
                                if outname_l != name_l-1:
                                    continue
                        else:
                            if '_LAYER' not in outname:
                                continue
                            idx = outname.find('_LAYER')
                            outname_l = int(outname[(idx+ len('_LAYER')):])
                            if outname_l != num_layers-2:
                                continue
                            else:
                                if '_FINAL' not in name:
                                    name += '_FINAL'
                        print ('edge: %s -> %s'%(name, outname))
                        H.add_edge(name, outname)
                H.layout()
                H.draw('/tmp/pyviz_rendered.png', prog='dot')
            from PIL import Image 
            image = Image.open('/tmp/pyviz_rendered.png')
            #fig['layout'] = layout
            img_data = np.asarray(image.convert('L'))
            window1 = vis.image(img_data)
            window2 = vis.plotlyplot(fig)
            #vis.update_window_opts(win=window, opts=dict(width=width, height=height))
            return window1, window2
        global callback_window1
        global callback_window2
        callback_window1, callback_window2 = draw_batch(cur_batch, (default_collapsed==1))
        


        #try:
        #    input = raw_input  # for Python 2 compatibility
        #except NameError:
        #    pass
        input('Waiting for callbacks, press enter to quit.')

if __name__ == '__main__':
    main(opts)
