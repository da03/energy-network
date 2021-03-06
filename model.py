import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math, copy, time
from optim_n2n import *
from torch.autograd import Variable
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

global flag
flag = False
global flag2
flag2 = False
def sample_gumbel(input):
    noise = torch.rand(input.size()).to(input)
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return noise

def gumbel_softmax_sample(log_probs, temperature):
    noise = sample_gumbel(log_probs)
    x = (log_probs + noise) / temperature
    x = F.softmax(x, dim=-1)
    return x.view_as(log_probs)

class EnergyNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(EnergyNetwork, self).__init__()
        self.linear_x = nn.Linear(d_model, d_ff)
        self.linear_y = nn.Linear(d_model, d_ff)
        
        self.out = nn.Sequential(nn.LeakyReLU(), nn.Linear(d_ff, 1))

    def forward(self, hx, hy):
        return self.out(self.linear_x(hx) + self.linear_y(hy)).view(-1).sum()

class Model(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, mode, encoder_fx, encoder_fy, decoder_gx, decoder_gy, init_x, init_y, src_embed, trg_embed, generator_gx, generator_gy, residual_var=0, selfmode='soft', encselfmode='soft', selfdependent_posterior=0):
        super(Model, self).__init__()
        self.mode = mode
        self.selfmode = selfmode
        self.encselfmode = encselfmode
        self.encoder_fx = encoder_fx
        self.encoder_fy = encoder_fy
        self.decoder_gx = decoder_gx
        self.decoder_gy = decoder_gy
        self.init_x = init_x
        self.init_y = init_y
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.trg_pad = None
        self.residual_var = residual_var
        self.generator_gx = generator_gx
        self.generator_gy = generator_gy
        self.selfdependent_posterior = selfdependent_posterior
                                                                                    
    def forward_gy_fy_y(self, src, trg, src_mask, trg_mask, src_in=None, src_out=None, trg_in=None, trg_out=None, src_mask_out=None, trg_mask_out=None, temperature=None, selftemperature=None, encselftemperature=None):
        assert self.trg_pad is not None
        self.samples = None
        trg_embeddings = self.trg_embed(trg)
        h, log_enc_self_attention_priors, enc_self_attention_priors, enc_self_attention_samples = self.encoder_fy(trg_embeddings, trg_mask)
        h = h.mean(1).view(h.size(0), 1, -1)
        if trg_mask_out is not None:
            trg_mask_out = trg_mask_out[:, :-1, :-1]
        trg_embeddings = trg_embeddings[:, :-1]
        decoder_output, log_prior_attentions, prior_attentions, log_self_attention_priors, self_attention_priors, self_attention_samples = self.decoder_gy(h, trg_embeddings, src_mask, trg_mask_out)
        self.self_attention_samples = self_attention_samples
        log_posterior_attentions, posterior_attentions = None, None
        return decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors

    def forward_gx_fx_x(self, src, trg, src_mask, trg_mask, src_in=None, src_out=None, trg_in=None, trg_out=None, src_mask_out=None, trg_mask_out=None, temperature=None, selftemperature=None, encselftemperature=None):
        assert self.trg_pad is not None
        self.samples = None
        src_embeddings = self.src_embed(src)
        h, log_enc_self_attention_priors, enc_self_attention_priors, enc_self_attention_samples = self.encoder_fx(src_embeddings, src_mask)
        h = h.mean(1).view(h.size(0), 1, -1)
        if src_mask_out is not None:
            src_mask_out = src_mask_out[:, :-1, :-1]
        src_embeddings = src_embeddings[:, :-1]
        decoder_output, log_prior_attentions, prior_attentions, log_self_attention_priors, self_attention_priors, self_attention_samples = self.decoder_gx(h, src_embeddings, src_mask, src_mask_out)
        self.self_attention_samples = self_attention_samples
        log_posterior_attentions, posterior_attentions = None, None
        return decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors
    def forward_gy_inity_fx_x(self, src, trg, src_mask, trg_mask, src_in=None, src_out=None, trg_in=None, trg_out=None, src_mask_out=None, trg_mask_out=None, temperature=None, selftemperature=None, encselftemperature=None):
        assert self.trg_pad is not None
        self.samples = None
        src_embeddings = self.src_embed(src)
        h, log_enc_self_attention_priors, enc_self_attention_priors, enc_self_attention_samples = self.encoder_fx(src_embeddings, src_mask)
        h = self.init_y(h.mean(1)).view(h.size(0), 1, -1)
        self.enc_self_attention_samples = enc_self_attention_samples
        trg_embeddings = self.trg_embed(trg_in)
        if trg_mask_out is not None:
            trg_mask_out = trg_mask_out[:, :-1, :-1]
        decoder_output, log_prior_attentions, prior_attentions, log_self_attention_priors, self_attention_priors, self_attention_samples = self.decoder_gy(h, trg_embeddings, src_mask, trg_mask_out)
        self.self_attention_samples = self_attention_samples
        log_posterior_attentions, posterior_attentions = None, None
        return decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors
    def forward_gx_initx_fy_y(self, src, trg, src_mask, trg_mask, src_in=None, src_out=None, trg_in=None, trg_out=None, src_mask_out=None, trg_mask_out=None, temperature=None, selftemperature=None, encselftemperature=None):
        assert self.trg_pad is not None
        self.samples = None
        trg_embeddings = self.src_embed(trg)
        h, log_enc_self_attention_priors, enc_self_attention_priors, enc_self_attention_samples = self.encoder_fx(trg_embeddings, trg_mask)
        h = self.init_x(h.mean(1)).view(h.size(0), 1, -1)
        self.enc_self_attention_samples = enc_self_attention_samples
        src_embeddings = self.src_embed(src_in)
        if src_mask_out is not None:
            src_mask_out = src_mask_out[:, :-1, :-1]
        decoder_output, log_prior_attentions, prior_attentions, log_self_attention_priors, self_attention_priors, self_attention_samples = self.decoder_gx(h, src_embeddings, src_mask, src_mask_out)
        self.self_attention_samples = self_attention_samples
        log_posterior_attentions, posterior_attentions = None, None
        return decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors

    def beam_search_soft(self, beam_size, src, src_mask, sos, eos, max_length):
        assert not self.training
        src_embeddings = self.src_embed(src)
        h, _, _, _ = self.encoder_fx(src_embeddings, src_mask)
        h = self.init_y(h.mean(1)).view(h.size(0), 1, h.size(-1))
        # soft always attn_dropout
        hypothesis, score = self.decoder_gy.beam_search_soft(beam_size, self.generator_gy, self.trg_embed, h, src_mask, sos, eos, max_length)
        return hypothesis, score
 
    def beam_search_soft_energy(self, eta, energy_network, unroll, beam_size, src, src_mask, sos, eos, max_length):
        assert not self.training
        src_embeddings_full = self.src_embed(src)
        hx_init, _, _, _= self.encoder_fx(src_embeddings_full, src_mask)
        hy_init_ = self.init_y(hx_init.mean(1)).view(hx_init.size(0), 1, -1)
        hy_init = hy_init_.data.clone()
        hy_init.requires_grad = True
        def loss_fn(input):
            hy = input[0]
            energy = energy_network(hx_init.detach(), hy)
            return energy
        
        update_params = list(energy_network.parameters())
        meta_optimizer = OptimN2N(loss_fn, update_params, eps = 1e-5, 
                                    lr = eta,
                                    iters = unroll, momentum = 0.5,
                                    acc_param_grads=True,  
                                    max_grad_norm = 5)
        if unroll == 0:
            hy = hy_init
        else:
            hy = meta_optimizer.forward([hy_init])[0]
        # soft always attn_dropout
        h = hy
        hypothesis, score = self.decoder_gy.beam_search_soft(beam_size, self.generator_gy, self.trg_embed, h, src_mask, sos, eos, max_length)
        return hypothesis, score
 
    def beam_search_hard(self, beam_size, src, src_mask, sos, eos, max_length,  encselftemperature=None, encselfdependent_posterior=0, encattn_dropout=True):
        assert not self.training
        src_embeddings = self.src_embed(src)
        h, _, _, _ = self.encoder(src_embeddings, src_mask, temperature=encselftemperature, dependent_posterior=encselfdependent_posterior, attn_dropout=encattn_dropout)
        # soft always attn_dropout
        hypothesis, score = self.decoder.beam_search_hard(beam_size, self.generator, self.trg_embed, h, src_mask, sos, eos, max_length, attn_dropout=False, dependent_posterior=1)
        return hypothesis, score
    def beam_search_soft_selfhard(self, beam_size, src, src_mask, sos, eos, max_length,  encselftemperature=None, encselfdependent_posterior=0, encattn_dropout=True):
        assert not self.training
        src_embeddings = self.src_embed(src)
        h, _, _, _ = self.encoder(src_embeddings, src_mask, temperature=encselftemperature, dependent_posterior=encselfdependent_posterior, attn_dropout=encattn_dropout)
        # soft always attn_dropout
        hypothesis, score = self.decoder.beam_search_soft_selfhard(beam_size, self.generator, self.trg_embed, h, src_mask, sos, eos, max_length, attn_dropout=True)
        return hypothesis, score
 
    def beam_search_hard_selfhard(self, beam_size, src, src_mask, sos, eos, max_length,  encselftemperature=None, encselfdependent_posterior=0, encattn_dropout=True):
        assert not self.training
        src_embeddings = self.src_embed(src)
        h, _, _, _ = self.encoder(src_embeddings, src_mask, temperature=encselftemperature, dependent_posterior=encselfdependent_posterior, attn_dropout=encattn_dropout)
        # soft always attn_dropout
        hypothesis, score = self.decoder.beam_search_hard_selfhard(beam_size, self.generator, self.trg_embed, h, src_mask, sos, eos, max_length, attn_dropout=False)
        return hypothesis, score


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
                                                
    def forward(self, x, mask, temperature=None, dependent_posterior=0, attn_dropout=True):
        log_enc_self_attention_priors, enc_self_attention_priors, enc_self_attention_samples = [], [], []
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, log_enc_self_attention_prior, enc_self_attention_prior, enc_self_attention_sample = layer(x, mask, temperature=temperature, dependent_posterior=dependent_posterior, attn_dropout=attn_dropout)
            log_enc_self_attention_priors.append(log_enc_self_attention_prior)
            enc_self_attention_priors.append(enc_self_attention_prior)
            enc_self_attention_samples.append(enc_self_attention_sample)
 
        return self.norm(x), log_enc_self_attention_priors, enc_self_attention_priors, enc_self_attention_samples

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer, num_outputs=1):
        "Apply residual connection to any sublayer with the same size."
        if num_outputs == 1:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            log_prior_attention, prior_attention, sample, z = sublayer(self.norm(x))
            return log_prior_attention, prior_attention, sample, x + self.dropout(z)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, temperature=None, dependent_posterior=0, attn_dropout=True):
        "Follow Figure 1 (left) for connections."
        #x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        log_self_attention_prior, self_attention_prior, self_attention_sample, x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask, num_outputs=2, dependent_posterior=dependent_posterior, temperature=temperature, attn_dropout=attn_dropout), num_outputs=2)
        return self.sublayer[1](x, self.feed_forward), log_self_attention_prior, self_attention_prior, self_attention_sample

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
                                               
    def forward(self, h, trg, src_mask, trg_mask=None):
        #import pdb; pdb.set_trace()
        z = trg + h
        prior_attentions = []
        log_prior_attentions = []
        log_self_attention_priors = []
        self_attention_priors = []
        self_attention_samples = []
        samples = []
        for layer in self.layers:
            log_prior_attention, prior_attention, sample, z, log_self_attention_prior, self_attention_prior, self_attention_sample = layer(z, h, src_mask, trg_mask)
            prior_attentions.append(prior_attention)
            log_prior_attentions.append(log_prior_attention)
            if sample is not None:
                samples.append(sample)
            log_self_attention_priors.append(log_self_attention_prior)
            self_attention_priors.append(self_attention_prior)
            if self_attention_sample is not None:
                self_attention_samples.append(self_attention_sample)
        z = self.norm(z)
        if len(samples) > 0:
            return z, log_prior_attentions, prior_attentions, samples, log_self_attention_priors, self_attention_priors, self_attention_samples
        else:
            return z, log_prior_attentions, prior_attentions, log_self_attention_priors, self_attention_priors, self_attention_samples


    def beam_search_soft(self, beam_size, generator, trg_embed, memory, src_mask, sos, eos, max_length):
        trg_embed[1].offset = 0
        batch_size, src_length, _ = memory.size()
        assert batch_size == 1
        trg = memory.new_full((1, 1), sos).long()
        trg_embeddings = trg_embed(trg) # 1, 1, hidden
        hiddens = [None for _ in self.layers]
        total_scores = None
        current_inputs = trg_embeddings
        best_finished_hypothesis = None
        best_finished_score = -float('inf')
        current_hypotheses = None
        for t in range(max_length):
            if memory.size(0) != trg_embeddings.size(0):
                memory = memory.expand(beam_size, -1, -1)
                src_mask = src_mask.expand(beam_size, -1, -1)
            z = trg_embeddings + memory
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z_selfattend = current_inputs 
                else:
                    z_selfattend = hiddens[i-1]
                log_prior_attention, prior_attention, sample, z, _, _, _ = layer(z, memory, src_mask, trg_mask=None, z_selfattend=z_selfattend)
                if hiddens[i] is None:
                    hiddens[i] = z
                else:
                    hiddens[i] = torch.cat([hiddens[i], z], 1)
            z = self.norm(z)
            decoder_output = z
            word_log_probs = generator(decoder_output).view(z.size(0), -1) # 1, V for first step
            vocab_size = word_log_probs.size(1)
            if total_scores is None:
                total_scores = word_log_probs # 1, v
            else:
                total_scores = total_scores.view(-1, 1) + word_log_probs # 5, v
            total_scores = total_scores.view(-1)
            beam_scores, beam_word_ids = total_scores.topk(beam_size, 0, sorted=True) # 5
            beam_word_ids = beam_word_ids.view(-1) % vocab_size
            total_scores = beam_scores.view(-1) # 5
            beam_from = beam_word_ids / vocab_size
            for i, hidden in enumerate(hiddens):
                hiddens[i] = hidden.gather(0, beam_from.view(-1, 1, 1).expand(-1, hidden.size(1), hidden.size(2)))
            beam_word_ids = beam_word_ids.view(-1, 1)
            #print (beam_word_ids)
            if current_hypotheses is not None:
                current_hypotheses = current_hypotheses.gather(0, beam_from.view(-1, 1).expand(-1, current_hypotheses.size(1)))
                current_hypotheses = torch.cat([current_hypotheses, beam_word_ids.view(-1, 1)], 1)
            else:
                current_hypotheses = beam_word_ids.view(-1, 1)
            current_inputs = current_inputs.gather(0, beam_from.view(-1, 1, 1).expand(-1, current_inputs.size(1), current_inputs.size(2)))
            trg_embed[1].offset = t + 1
            trg_embeddings = trg_embed(beam_word_ids) # 5, 1, hidden
            current_inputs = torch.cat([current_inputs, trg_embeddings], 1) # 5, 2, hidden
            for k in range(beam_size):
                if current_hypotheses[k][-1] == eos:
                    score = total_scores[k].item()
                    if score > best_finished_score:
                        best_finished_score = score
                        best_finished_hypothesis = current_hypotheses[k].data.clone()
                    total_scores[k] = -float('inf')
                    if k == 0:
                        break

        trg_embed[1].offset = 0
        return best_finished_hypothesis if best_finished_hypothesis is not None else current_hypotheses[0], best_finished_score if best_finished_hypothesis is not None else total_scores[0]

    def beam_search_soft_selfhard(self, beam_size, generator, trg_embed, memory, src_mask, sos, eos, max_length, attn_dropout=True):
        trg_embed[1].offset = 0
        batch_size, src_length, _ = memory.size()
        assert batch_size == 1
        trg = memory.new_full((1, 1), sos).long()
        trg_embeddings = trg_embed(trg) # 1, 1, hidden
        hiddens = [None for _ in self.layers]
        total_scores = None
        current_inputs = trg_embeddings
        best_finished_hypothesis = None
        best_finished_score = -float('inf')
        current_hypotheses = None
        for t in range(max_length):
            z = trg_embeddings
            if memory.size(0) != z.size(0):
                memory = memory.expand(beam_size, -1, -1)
                src_mask = src_mask.expand(beam_size, -1, -1)
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z_selfattend = current_inputs 
                else:
                    z_selfattend = hiddens[i-1]
                log_prior_attention, prior_attention, sample, z, _, _, _ = layer(z, memory, src_mask, trg_mask=None, attn_dropout=attn_dropout, z_selfattend=z_selfattend, selftemperature=-1)
                if hiddens[i] is None:
                    hiddens[i] = z
                else:
                    hiddens[i] = torch.cat([hiddens[i], z], 1)
            z = self.norm(z)
            decoder_output = z
            word_log_probs = generator(decoder_output).view(z.size(0), -1) # 1, V for first step
            vocab_size = word_log_probs.size(1)
            if total_scores is None:
                total_scores = word_log_probs # 1, v
            else:
                total_scores = total_scores.view(-1, 1) + word_log_probs # 5, v
            total_scores = total_scores.view(-1)
            beam_scores, beam_word_ids = total_scores.topk(beam_size, 0, sorted=True) # 5
            beam_word_ids = beam_word_ids.view(-1) % vocab_size
            total_scores = beam_scores.view(-1) # 5
            beam_from = beam_word_ids / vocab_size
            for i, hidden in enumerate(hiddens):
                hiddens[i] = hidden.gather(0, beam_from.view(-1, 1, 1).expand(-1, hidden.size(1), hidden.size(2)))
            beam_word_ids = beam_word_ids.view(-1, 1)
            #print (beam_word_ids)
            if current_hypotheses is not None:
                current_hypotheses = current_hypotheses.gather(0, beam_from.view(-1, 1).expand(-1, current_hypotheses.size(1)))
                current_hypotheses = torch.cat([current_hypotheses, beam_word_ids.view(-1, 1)], 1)
            else:
                current_hypotheses = beam_word_ids.view(-1, 1)
            current_inputs = current_inputs.gather(0, beam_from.view(-1, 1, 1).expand(-1, current_inputs.size(1), current_inputs.size(2)))
            trg_embed[1].offset = t + 1
            trg_embeddings = trg_embed(beam_word_ids) # 5, 1, hidden
            current_inputs = torch.cat([current_inputs, trg_embeddings], 1) # 5, 2, hidden
            for k in range(beam_size):
                if current_hypotheses[k][-1] == eos:
                    score = total_scores[k].item()
                    if score > best_finished_score:
                        best_finished_score = score
                        best_finished_hypothesis = current_hypotheses[k].data.clone()
                    total_scores[k] = -float('inf')
                    if k == 0:
                        break

        trg_embed[1].offset = 0
        return best_finished_hypothesis if best_finished_hypothesis is not None else current_hypotheses[0], best_finished_score if best_finished_hypothesis is not None else total_scores[0]

    def beam_search_hard(self, beam_size, generator, trg_embed, memory, src_mask, sos, eos, max_length, attn_dropout=True, dependent_posterior=1):
        trg_embed[1].offset = 0
        batch_size, src_length, _ = memory.size()
        assert batch_size == 1
        trg = memory.new_full((1, 1), sos).long()
        trg_embeddings = trg_embed(trg) # 1, 1, hidden
        hiddens = [None for _ in self.layers]
        total_scores = None
        current_inputs = trg_embeddings
        best_finished_hypothesis = None
        best_finished_score = -float('inf')
        current_hypotheses = None
        for t in range(max_length):
            z = trg_embeddings
            if memory.size(0) != z.size(0):
                memory = memory.expand(beam_size, -1, -1)
                src_mask = src_mask.expand(beam_size, -1, -1)
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z_selfattend = current_inputs 
                else:
                    z_selfattend = hiddens[i-1]
                log_prior_attention, prior_attention, sample, z, _, _, _ = layer(z, memory, src_mask, trg_mask=None, attn_dropout=attn_dropout, z_selfattend=z_selfattend, temperature=-1, dependent_posterior=dependent_posterior)
                if hiddens[i] is None:
                    hiddens[i] = z
                else:
                    hiddens[i] = torch.cat([hiddens[i], z], 1)
            z = self.norm(z)
            decoder_output = z
            word_log_probs = generator(decoder_output).view(z.size(0), -1) # 1, V for first step
            vocab_size = word_log_probs.size(1)
            if total_scores is None:
                total_scores = word_log_probs # 1, v
            else:
                total_scores = total_scores.view(-1, 1) + word_log_probs # 5, v
            total_scores = total_scores.view(-1)
            beam_scores, beam_word_ids = total_scores.topk(beam_size, 0, sorted=True) # 5
            beam_word_ids = beam_word_ids.view(-1) % vocab_size
            total_scores = beam_scores.view(-1) # 5
            beam_from = beam_word_ids / vocab_size
            for i, hidden in enumerate(hiddens):
                hiddens[i] = hidden.gather(0, beam_from.view(-1, 1, 1).expand(-1, hidden.size(1), hidden.size(2)))
            beam_word_ids = beam_word_ids.view(-1, 1)
            #print (beam_word_ids)
            if current_hypotheses is not None:
                current_hypotheses = current_hypotheses.gather(0, beam_from.view(-1, 1).expand(-1, current_hypotheses.size(1)))
                current_hypotheses = torch.cat([current_hypotheses, beam_word_ids.view(-1, 1)], 1)
            else:
                current_hypotheses = beam_word_ids.view(-1, 1)
            current_inputs = current_inputs.gather(0, beam_from.view(-1, 1, 1).expand(-1, current_inputs.size(1), current_inputs.size(2)))
            trg_embed[1].offset = t + 1
            trg_embeddings = trg_embed(beam_word_ids) # 5, 1, hidden
            current_inputs = torch.cat([current_inputs, trg_embeddings], 1) # 5, 2, hidden
            for k in range(beam_size):
                if current_hypotheses[k][-1] == eos:
                    score = total_scores[k].item()
                    if score > best_finished_score:
                        best_finished_score = score
                        best_finished_hypothesis = current_hypotheses[k].data.clone()
                    total_scores[k] = -float('inf')
                    if k == 0:
                        break

        trg_embed[1].offset = 0
        return best_finished_hypothesis if best_finished_hypothesis is not None else current_hypotheses[0], best_finished_score if best_finished_hypothesis is not None else total_scores[0]

    def beam_search_hard_selfhard(self, beam_size, generator, trg_embed, memory, src_mask, sos, eos, max_length, attn_dropout=True):
        trg_embed[1].offset = 0
        batch_size, src_length, _ = memory.size()
        assert batch_size == 1
        trg = memory.new_full((1, 1), sos).long()
        trg_embeddings = trg_embed(trg) # 1, 1, hidden
        hiddens = [None for _ in self.layers]
        total_scores = None
        current_inputs = trg_embeddings
        best_finished_hypothesis = None
        best_finished_score = -float('inf')
        current_hypotheses = None
        for t in range(max_length):
            z = trg_embeddings
            if memory.size(0) != z.size(0):
                memory = memory.expand(beam_size, -1, -1)
                src_mask = src_mask.expand(beam_size, -1, -1)
            for i, layer in enumerate(self.layers):
                if i == 0:
                    z_selfattend = current_inputs 
                else:
                    z_selfattend = hiddens[i-1]
                log_prior_attention, prior_attention, sample, z, _, _, _ = layer(z, memory, src_mask, trg_mask=None, attn_dropout=attn_dropout, z_selfattend=z_selfattend, temperature=-1, selftemperature=-1, dependent_posterior=1, selfdependent_posterior=1)
                if hiddens[i] is None:
                    hiddens[i] = z
                else:
                    hiddens[i] = torch.cat([hiddens[i], z], 1)
            z = self.norm(z)
            decoder_output = z
            word_log_probs = generator(decoder_output).view(z.size(0), -1) # 1, V for first step
            vocab_size = word_log_probs.size(1)
            if total_scores is None:
                total_scores = word_log_probs # 1, v
            else:
                total_scores = total_scores.view(-1, 1) + word_log_probs # 5, v
            total_scores = total_scores.view(-1)
            beam_scores, beam_word_ids = total_scores.topk(beam_size, 0, sorted=True) # 5
            beam_word_ids = beam_word_ids.view(-1) % vocab_size
            total_scores = beam_scores.view(-1) # 5
            beam_from = beam_word_ids / vocab_size
            for i, hidden in enumerate(hiddens):
                hiddens[i] = hidden.gather(0, beam_from.view(-1, 1, 1).expand(-1, hidden.size(1), hidden.size(2)))
            beam_word_ids = beam_word_ids.view(-1, 1)
            #print (beam_word_ids)
            if current_hypotheses is not None:
                current_hypotheses = current_hypotheses.gather(0, beam_from.view(-1, 1).expand(-1, current_hypotheses.size(1)))
                current_hypotheses = torch.cat([current_hypotheses, beam_word_ids.view(-1, 1)], 1)
            else:
                current_hypotheses = beam_word_ids.view(-1, 1)
            current_inputs = current_inputs.gather(0, beam_from.view(-1, 1, 1).expand(-1, current_inputs.size(1), current_inputs.size(2)))
            trg_embed[1].offset = t + 1
            trg_embeddings = trg_embed(beam_word_ids) # 5, 1, hidden
            current_inputs = torch.cat([current_inputs, trg_embeddings], 1) # 5, 2, hidden
            for k in range(beam_size):
                if current_hypotheses[k][-1] == eos:
                    score = total_scores[k].item()
                    if score > best_finished_score:
                        best_finished_score = score
                        best_finished_hypothesis = current_hypotheses[k].data.clone()
                    total_scores[k] = -float('inf')
                    if k == 0:
                        break

        trg_embed[1].offset = 0
        return best_finished_hypothesis if best_finished_hypothesis is not None else current_hypotheses[0], best_finished_score if best_finished_hypothesis is not None else total_scores[0]


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
                                                                 
    def forward(self, x, h, src_mask, trg_mask=None, z_selfattend=None):
        "Follow Figure 1 (right) for connections."
        if z_selfattend is None:
            z_selfattend = x
        #if trg_mask is None:
        #    x = self.sublayer[0](x, lambda x: self.self_attn(x, z_selfattend, z_selfattend))
        #else:
        #    x = self.sublayer[0](x, lambda x: self.self_attn(x, z_selfattend, z_selfattend, trg_mask))
        log_self_attention_prior, self_attention_prior, self_attention_sample, x = self.sublayer[0](x, lambda x: self.self_attn(x, z_selfattend, z_selfattend, trg_mask, num_outputs=2), num_outputs=2)
        #log_prior_attention, prior_attention, sample, x = self.sublayer[1](x, lambda x: self.src_attn(x, h, h, src_mask, num_outputs=2), num_outputs=2)
        return None, None, None, self.sublayer[2](x, self.feed_forward), log_self_attention_prior, self_attention_prior, self_attention_sample

class InferenceNetwork(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, dependent_posterior):
        super(InferenceNetwork, self).__init__()
        self.layers = clones(layer, N)
        self.dependent_posterior = dependent_posterior
                                               
    # maybe consider disallowing attending to oneself
    def forward(self, h, trg, src_mask, temperature, trg_mask=None, src=None):
        log_posterior_attentions = []
        posterior_attentions = []
        z = trg
        samples = []
        for layer in self.layers:
            global flag
            flag = True
            log_posterior_attention, posterior_attention, sample, z = layer(z, h, src_mask, dependent_posterior=self.dependent_posterior, temperature=temperature)
            samples.append(sample)
            log_posterior_attentions.append(log_posterior_attention)
            posterior_attentions.append(posterior_attention)
        return log_posterior_attentions, posterior_attentions, samples

class LSTMInferenceNetwork(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, W, N, sharelstm=0):
        super(LSTMInferenceNetwork, self).__init__()
        self.layers_src = clones(layer, N)
        self.layers_trg = clones(layer, N)
       
        self.Ws = clones(W, N)
        self.dependent_posterior = 0
        self.sharelstm = sharelstm
        print ('sharelstm: %d'%sharelstm)
                                               
    # maybe consider disallowing attending to oneself
    def forward(self, h, trg, src_mask, temperature, trg_mask, src):
        log_posterior_attentions = []
        posterior_attentions = []
        trg_lengths = trg_mask.sum(-1).contiguous().view(-1)
        src_lengths = src_mask.sum(-1).contiguous().view(-1).cpu().numpy()
        samples = []
        src = src.transpose(0, 1)
        # pack them up nicely
        trg_lengths_s, perm_index = trg_lengths.sort(0, descending=True)
        trg = trg[perm_index]
        trg = trg.transpose(0, 1)
        packed_trg = pack_padded_sequence(trg, trg_lengths_s.cpu().numpy())

        packed_src = pack_padded_sequence(src, src_lengths)

        s1 = None
        s2 = None
        for src_layer, trg_layer, W in zip(self.layers_src, self.layers_trg, self.Ws):
            if s1 is None:
                s1 = src_layer
            if s2 is None:
                s2 = trg_layer
            if self.sharelstm != 0:
                src_layer = s1
                trg_layer = s2
            #print (z.size()) # batch_size, trg_len, hidden
            packed_trg_, _ = trg_layer(packed_trg)
            trg_outputs = pad_packed_sequence(packed_trg_)[0]
            packed_src_, _ = src_layer(packed_src)
            src_outputs = pad_packed_sequence(packed_src_)[0]
            src_memory_bank = src_outputs.transpose(0, 1)
            src_memory_bank = src_memory_bank.transpose(1, 2) # bsz, rnn, src_len


            bsz = src_memory_bank.size(0)
            rnn = src_memory_bank.size(1)
            src_len = src_memory_bank.size(-1)
            h = self.h
            src_memory_bank = src_memory_bank.contiguous().view(bsz, self.h, rnn//h, -1).contiguous().view(-1, rnn//h, src_len)
            trg_outputs = trg_outputs.transpose(0, 1)
            odx = perm_index.contiguous().view(-1, 1).unsqueeze(1).expand(trg_outputs.size(0), trg_outputs.size(1), trg_outputs.size(2))
            trg_outputs = trg_outputs.gather(0, odx)
            trg_memory_bank = W(trg_outputs) # bsz, trg_len, rnn_size
            trg_memory_bank = trg_memory_bank.transpose(1, 2)
            trg_len = trg_memory_bank.size(-1)
            trg_memory_bank = trg_memory_bank.contiguous().view(bsz, self.h, rnn//h, -1).contiguous().view(-1, rnn//h, trg_len).transpose(1,2)

            scores = torch.bmm(trg_memory_bank, src_memory_bank).contiguous().view(bsz, self.h, trg_len, src_len)
            scores.data.masked_fill_(src_mask.unsqueeze(1).expand(bsz, self.h, trg_len, src_len)==0, -1e9)
            p_attn = F.softmax(scores, dim = -1)
            log_p_attn = F.log_softmax(scores, dim = -1)


            assert temperature is not None
            if temperature > 0:
                sample = gumbel_softmax_sample(log_p_attn, temperature)
            else: # true categorical sample, equivalently we can use argmax to replace softmax in gumbel, here we use categorical for sanity check.
                h1, h2, h3, h4 = p_attn.size()
                p = Categorical(p_attn.view(-1, h4))
                sample_id = p.sample().view(h1, h2, h3, 1)
                sample = p_attn.new_full(p_attn.size(), 0.).to(p_attn)
                sample.scatter_(3, sample_id, 1.)
            log_posterior_attentions.append(log_p_attn)
            posterior_attentions.append(p_attn)
            samples.append(sample)
        return log_posterior_attentions, posterior_attentions, samples

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None, attn=None, attn_dropout=True, dependent_posterior=0, temperature=None):
    if temperature is not None:
        assert dependent_posterior == 1
    "Compute 'Scaled Dot Product Attention'"
    global flag2
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
    / math.sqrt(d_k)
    if mask is not None:
        scores.data.masked_fill_(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    log_p_attn = F.log_softmax(scores, dim = -1)
    if attn_dropout and (dropout is not None):
        p_attn = dropout(p_attn)
    if attn is None:
        if temperature is None:
            return torch.matmul(p_attn, value), log_p_attn, p_attn, None
        if temperature > 0:
            sample = gumbel_softmax_sample(log_p_attn, temperature)
        elif temperature < 0:
            h1, h2, h3, h4 = p_attn.size()
            #p = Categorical(p_attn.view(-1, h4))
            _, sample_id = torch.max(p_attn.view(-1, h4), -1)
            #sample_id = p.sample().view(h1, h2, h3, 1)
            sample_id = sample_id.view(h1, h2, h3, 1)
            sample = p_attn.new_full(p_attn.size(), 0.).to(p_attn)
            sample.scatter_(3, sample_id, 1.)
        else: # true categorical sample, equivalently we can use argmax to replace softmax in gumbel, here we use categorical for sanity check.
            h1, h2, h3, h4 = p_attn.size()
            p = Categorical(p_attn.view(-1, h4))
            sample_id = p.sample().view(h1, h2, h3, 1)
            sample = p_attn.new_full(p_attn.size(), 0.).to(p_attn)
            sample.scatter_(3, sample_id, 1.)
        if dependent_posterior == 0:
            return torch.matmul(p_attn, value), log_p_attn, p_attn, sample
        else:
            return torch.matmul(sample, value), log_p_attn, p_attn, sample
    else:
        return torch.matmul(attn, value), log_p_attn, p_attn, None

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
                                                                                            
    def forward(self, query, key, value, mask=None, num_outputs=1, attn=None, attn_dropout=True, dependent_posterior=0, temperature=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
                                                                                                                                                                
         # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
        [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        for l, x in zip(self.linears, (query, key, value))]
      
        # 2) Apply attention on all the projected vectors in batch. 
        x, log_attn, self.attn, sample = attention(query, key, value, mask=mask, 
        dropout=self.dropout, attn=attn, attn_dropout=attn_dropout, dependent_posterior=dependent_posterior, temperature=temperature)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
        .view(nbatches, -1, self.h * self.d_k)
        if num_outputs == 1:
            return self.linears[-1](x)
        else:
            return log_attn, self.attn, sample, self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
                                            
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.offset = 0
                                                                                                                    
    def forward(self, x):
        #print ('forwarded: %d'%self.offset)
        x = x + Variable(self.pe[:, self.offset:(self.offset+x.size(1))], 
        requires_grad=False)
        return self.dropout(x)


def make_model(mode, src_vocab, trg_vocab, n_enc=6, n_dec=6,
                       d_model=512, d_ff=2048, h=8, dropout=0.1, share_decoder_embeddings=0,
                       share_word_embeddings=0, dependent_posterior=0, sharelstm=0, residual_var=0, selfmode='soft', encselfmode='soft', selfdependent_posterior=0):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)


    encoder_fx = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_enc)
    encoder_fy = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_enc)
    src_emb = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    decoder_gx = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_dec)
    decoder_gy = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), n_dec)
    init_x = nn.Sequential(nn.Linear(d_model, d_ff), nn.LeakyReLU(), nn.Linear(d_ff, d_model))
    init_y = nn.Sequential(nn.Linear(d_model, d_ff), nn.LeakyReLU(), nn.Linear(d_ff, d_model))
    trg_emb = nn.Sequential(Embeddings(d_model, trg_vocab), c(position))
    generator_gx = Generator(d_model, src_vocab)
    generator_gy = Generator(d_model, trg_vocab)
    model = Model(
                  mode,
                  encoder_fx,
                  encoder_fy,
                  decoder_gx,
                  decoder_gy,
                  init_x,
                  init_y,
                  src_emb,
                  trg_emb,
                  generator_gx, generator_gy, residual_var=residual_var, selfmode=selfmode, encselfmode=encselfmode, selfdependent_posterior=selfdependent_posterior)

    if share_decoder_embeddings == 1:
        model.generator_gy.proj.weight = model.trg_embed[0].lut.weight
        model.generator_gx.proj.weight = model.src_embed[0].lut.weight
        #model.generator.proj.bias = None
    #if share_word_embeddings == 1:
    #    model.src_embed[0].lut.weight = model.trg_embed[0].lut.weight
                                
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for n,p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        if 'lstm' in n:
            p.data.uniform_(-0.05, 0.05)
    return model
