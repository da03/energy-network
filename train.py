import argparse
import sys
import math

import torch
import torchtext
import torch.optim as optim

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
                    choices=["soft", "var", "hard", "lstmvar"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--selfmode", required=True,
                    choices=["soft", "var", "hard", "lstmvar"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--encselfmode", required=True,
                    choices=["soft", "var", "hard", "lstmvar"],  help="Training model. Default to be vanilla transformer with soft attention.")
#TODO: 
parser.add_argument("--xx", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--yy", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--yx", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--xy", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--share_decoder_embeddings", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--dependent_posterior", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--selfdependent_posterior", default=1, type=int,
                    choices=[1, 0],  help="Share decoder embeddings or not.")
parser.add_argument("--share_word_embeddings", default=0,
                    choices=[1, 0],  help="Share src trg embeddings or not.")
parser.add_argument("--heads", type=int, default=2, help="Number of tokens per minibatch")
parser.add_argument("--fix_model_steps", type=int, default=-1, help="Maximum Src Length")
parser.add_argument("--max_src_len", type=int, default=150, help="Maximum Src Length")
parser.add_argument("--max_trg_len", type=int, default=150, help="Maximum Trg Length")
parser.add_argument("--batch_size", type=int, default=3200, help="Number of tokens per minibatch")
parser.add_argument("--accum_grad", type=int, default=1, help="Number of tokens per minibatch")
parser.add_argument("--epochs", type=int, default=50, help="Number of Epochs")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning Rate")
parser.add_argument("--temperature", type=float, required=True, help="Gumbel Softmax temperature")
parser.add_argument("--dropout", type=float, default=0.1, help="Gumbel Softmax temperature")
parser.add_argument("--lr_begin", type=float, default=3e-4, help="Gumbel Softmax temperature")
parser.add_argument("--lr_end", type=float, default=1e-5, help="Gumbel Softmax temperature")
parser.add_argument("--anneal_steps", type=float, default=250000, help="Gumbel Softmax temperature")
parser.add_argument("--selftemperature", type=float, required=True, help="Gumbel Softmax temperature")
parser.add_argument("--encselftemperature", type=float, required=True, help="Gumbel Softmax temperature")
parser.add_argument("--anneal_temperature", type=int, default=0, help="Gumbel Softmax temperature")
parser.add_argument("--anneal_kl", type=int, default=1, help="Gumbel Softmax temperature")
parser.add_argument("--residual_var", type=int, default=0, help="Gumbel Softmax temperature")
parser.add_argument("--sharelstm", type=int, default=0, help="Gumbel Softmax temperature")
parser.add_argument("--train_from", default="", help="Model Path")
parser.add_argument("--save_to", type=str, required=True, help="Model Path")
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
        exts = ['.en', '.de']
    elif opts.direction == 'deen':
        exts = ['.de', '.en']
    else:
        raise NotImplementedError


    SRC = torchtext.data.Field(
                pad_token=PAD_WORD,
            init_token=BOS_WORD, eos_token=EOS_WORD,
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

    def dyn_batch_without_padding(new, i, sofar):
        if not hasattr(new, 'src'):
            new_src = 0
        else:
            new_src = len(new.src)
        return sofar + max(new_src, len(new.trg))

    def batch_size_fn_with_padding(new, count, sofar):
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
        if not hasattr(new, 'src'):
            new_src = 0
        else:
            new_src = len(new.src)
        # Src: <bos> w1 ... wN <eos>
        max_src_in_batch = max(max_src_in_batch, new_src + 2)
        # Tgt: w1 ... wN <eos>
        max_trg_in_batch = max(max_trg_in_batch, len(new.trg) + 1)
        src_elements = count * max_src_in_batch
        trg_elements = count * max_trg_in_batch
        return max(src_elements, trg_elements)

    batch_size_fn = dyn_batch_without_padding
    batch_size_fn = batch_size_fn_with_padding

    BATCH_SIZE = opts.batch_size
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)) if hasattr(x, 'src') else len(x.trg),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True, sort_within_batch=True)
    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)) if hasattr(x, 'src') else len(x.trg),
                            batch_size_fn=batch_size_fn, train=False)
    TRG.build_vocab(train.src, train.trg)
    SRC.vocab = TRG.vocab

    # Build Model
    print (' '.join(sys.argv))
    trg_vocab_size = len(TRG.vocab.itos)
    src_vocab_size = len(SRC.vocab.itos)
    print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    print ('Building Model')
    model = make_model(opts.mode, src_vocab_size, trg_vocab_size, n_enc=5, n_dec=5,
                   d_model=278, d_ff=507, h=opts.heads, dropout=opts.dropout, share_decoder_embeddings=opts.share_decoder_embeddings,
                   share_word_embeddings=opts.share_word_embeddings, dependent_posterior=opts.dependent_posterior,
                   sharelstm=opts.sharelstm, residual_var=opts.residual_var, selfmode = opts.selfmode, encselfmode=opts.encselfmode, selfdependent_posterior=opts.selfdependent_posterior)
    print (model)
    if opts.train_from != '':
        print ('Loading Model from %s'%opts.train_from)
        checkpoint = torch.load(opts.train_from)
        model.load_state_dict(checkpoint['model'])
    model.cuda()
    opts.lr_begin = opts.lr_begin / 2
    optimizer = get_std_opt(model, 746, model.trg_embed[0].d_model, opts.lr_begin, opts.lr_end, opts.anneal_steps)
    devices = [0]
    #devices = [0, 1, 2, 3]
    print (opts)
    model.trg_pad = TRG.vocab.stoi["<blank>"]
    model_par = nn.DataParallel(model, device_ids=devices)

    weight = torch.ones(trg_vocab_size)
    weight[TRG.vocab.stoi["<blank>"]] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    #criterion = LabelSmoothing(size=trg_vocab_size, padding_idx=TRG.vocab.stoi["<blank>"], smoothing=0.1)
    criterion.cuda()
                                   
    total_loss_yx = 0.
    total_loss_xx = 0.
    total_loss_yy = 0.
    total_loss_xy = 0.
    total_tokens_yx = 1e-9
    total_tokens_xx = 1e-9
    total_tokens_xy = 1e-9
    total_tokens_yy = 1e-9
    total_nonpadding_yx = 1e-9
    total_nonpadding_xx = 1e-9
    total_nonpadding_xy = 1e-9
    total_nonpadding_yy = 1e-9
    total_correct_yx = 0
    total_correct_xx = 0
    total_correct_xy = 0
    total_correct_yy = 0
    loss_all_yx = 0.
    loss_all_xx = 0.
    loss_all_xy = 0.
    loss_all_yy = 0.
    tokens_yx = 1e-9
    tokens_xx = 1e-9
    tokens_xy = 1e-9
    tokens_yy = 1e-9
    num_steps = 0
    opts.accum_grad = opts.xy + opts.yx + opts.xx + opts.yy
    loss_compute_gy = MultiGPULossCompute(opts.mode, TRG, model.generator_gy, criterion, devices=devices, optimizer=optimizer, model=model, anneal_kl=opts.anneal_kl)
    loss_compute_gy.alpha = 0
    loss_compute_gy.accum_grad = opts.accum_grad
    loss_compute_gx = MultiGPULossCompute(opts.mode, SRC, model.generator_gx, criterion, devices=devices, optimizer=optimizer, model=model, anneal_kl=opts.anneal_kl)
    loss_compute_gx.alpha = 0
    loss_compute_gx.accum_grad = opts.accum_grad
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
    global_steps = 0
    for epoch in range(opts.epochs):
        print ('')
        print ('Epoch: %d' %epoch)
        print ('Training')
        model.train()
        start = time.time()
        for i, batch in enumerate(train_iter):
            if global_steps <= opts.fix_model_steps:
                if global_steps == 0:
                    print ('Fixing model')
                    modules = [model.encoder, model.decoder, model.src_embed, model.trg_embed, model.generator]
                    for module in modules:
                        if module is None:
                            continue
                        for parameter in module.parameters():
                            parameter.requires_grad = False
                    #print ('Fixing loss')
                    #loss_compute.t_ = False
                    loss_compute.t_ = True
                if global_steps == opts.fix_model_steps:
                    print ('Training model')
                    modules = [model.encoder, model.decoder, model.src_embed, model.trg_embed, model.generator]
                    for module in modules:
                        if module is None:
                            continue
                        for parameter in module.parameters():
                            parameter.requires_grad = True
                    loss_compute.t_ = True
            global_steps += 1
            num_steps += 1
            src = batch.src[0].transpose(0, 1) # batch, len
            src_in = src[:, :-1]
            src_out = src[:, 1:]
            src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
            src_mask_out = src_mask & Variable(subsequent_mask(src.size(-1)).to(src_mask))
            trg = batch.trg.transpose(0, 1) # batch, len
            trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
            trg_mask_out = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
            trg_in = trg[:, :-1]
            trg_out = trg[:, 1:]
            if opts.anneal_temperature == 1:
                temperature = 1. - min(1.0, math.floor(epoch/5) * 1.0 / 8) * (1.-opts.temperature)
            else:
                temperature = opts.temperature
            selftemperature = opts.selftemperature
            encselftemperature = opts.encselftemperature

# auto encoding loss
            if opts.yy == 1:
                with torch.cuda.device(0):
                    decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gy_fy_y(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                    ntokens_trg = (trg_out != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                    loss_compute_gy.pad = TRG.vocab.stoi["<blank>"]
                    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gy(decoder_output, trg_out, ntokens_trg, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                    total_loss_yy += l
                    total_correct_yy += l_correct
                    total_nonpadding_yy += l_nonpadding
                    loss_all_yy += l
                    total_tokens_yy += ntokens_trg
                    tokens_yy += ntokens_trg

# auto encoding loss
            if opts.xx == 1:
                with torch.cuda.device(0):
                    decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gx_fx_x(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                    ntokens_src = (src_out != SRC.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                    loss_compute_gx.pad = SRC.vocab.stoi["<blank>"]
                    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gx(decoder_output, src_out, ntokens_src, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                    total_loss_xx += l
                    total_correct_xx += l_correct
                    total_nonpadding_xx += l_nonpadding
                    loss_all_xx += l
                    total_tokens_xx += ntokens_trg
                    tokens_xx += ntokens_trg

            if opts.yx == 1:
                with torch.cuda.device(0):
                    decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gy_inity_fx_x(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                    ntokens_trg = (trg_out != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                    loss_compute_gy.pad = TRG.vocab.stoi["<blank>"]
                    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gy(decoder_output, trg_out, ntokens_trg, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                    total_loss_yx += l
                    total_correct_yx += l_correct
                    total_nonpadding_yx += l_nonpadding
                    loss_all_yx += l
                    total_tokens_yx += ntokens_trg
                    tokens_yx += ntokens_trg
            if opts.xy == 1:
                with torch.cuda.device(0):
                    decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gx_initx_fy_y(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                    ntokens_src = (src_out != SRC.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                    loss_compute_gx.pad = SRC.vocab.stoi["<blank>"]
                    l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gx(decoder_output, src_out, ntokens_src, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                    total_loss_xy += l
                    total_correct_xy += l_correct
                    total_nonpadding_xy += l_nonpadding
                    loss_all_xy += l
                    total_tokens_xy += ntokens_src
                    tokens_xy += ntokens_src
            if i % 50 == 1:
                elapsed = max(0.1, time.time() - start)
                print("Epoch Step: %d lr: %f, PPL yx: %f, Acc yx: %f, PPL xy: %f, Acc xy: %f, PPL xx: %f, Acc xx: %f, PPL yy: %f, Acc yy: %f , Tokens per Sec: %f" %
                (i, loss_compute_gy.optimizer._rate, math.exp(min(100, loss_all_yx / tokens_yx)), float(total_correct_yx)/total_nonpadding_yx, math.exp(min(100, loss_all_xy / tokens_xy)), float(total_correct_xy)/total_nonpadding_xy, math.exp(min(100, loss_all_xx / tokens_xx)), float(total_correct_xx)/total_nonpadding_xx, math.exp(min(100, loss_all_yy / tokens_yy)), float(total_correct_yy)/total_nonpadding_yy, tokens_yy / elapsed))
                sys.stdout.flush()
                start = time.time()
                tokens_yy = 1e-9
                loss_all_yy = 0.
                total_correct_yy = 0
                total_nonpadding_yy = 1e-9
                tokens_yx = 1e-9
                loss_all_yx = 0.
                total_correct_yx = 0
                total_nonpadding_yx = 1e-9
                tokens_xx = 1e-9
                loss_all_xx = 0.
                total_correct_xx = 0
                total_nonpadding_xx = 1e-9
                tokens_xy = 1e-9
                loss_all_xy = 0.
                total_correct_xy = 0
                total_nonpadding_xy = 1e-9

        print ('Validation')
        model.eval()
        val_loss_all_yx = 0
        val_tokens_yx = 1e-9
        val_total_correct_yx = 0
        val_total_nonpadding_yx = 1e-9
        val_loss_all_xx = 0
        val_tokens_xx = 1e-9
        val_total_correct_xx = 0
        val_total_nonpadding_xx = 1e-9
        val_loss_all_xy = 0
        val_tokens_xy = 1e-9
        val_total_correct_xy = 0
        val_total_nonpadding_xy = 1e-9
        val_loss_all_yy = 0
        val_tokens_yy = 1e-9
        val_total_correct_yy = 0
        val_total_nonpadding_yy = 1e-9
        with torch.no_grad():
            for i, batch in enumerate(val_iter):
                num_steps += 1
                src = batch.src[0].transpose(0, 1) # batch, len
                src_in = src[:, :-1]
                src_out = src[:, 1:]
                src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
                src_mask_out = src_mask & Variable(subsequent_mask(src.size(-1)).to(src_mask))
                trg = batch.trg.transpose(0, 1) # batch, len
                trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
                trg_mask_out = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask_out))
                trg_in = trg[:, :-1]
                trg_out = trg[:, 1:]
                #decoder_output, prior_attentions, posterior_attentions = model_par(src, trg, src_mask, trg_mask)
# auto encod    ing loss
                if opts.yy == 1:
                    with torch.cuda.device(0):
                        optimizer = loss_compute_gy.optimizer
                        loss_compute_gy.optimizer = None
                        decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gy_fy_y(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                        ntokens_trg = (trg_out != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                        loss_compute_gy.pad = TRG.vocab.stoi["<blank>"]
                        l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gy(decoder_output, trg_out, ntokens_trg, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                        loss_compute_gy.optimizer = optimizer
                        val_loss_all_yy += l
                        val_tokens_yy += ntokens_trg
                        val_total_correct_yy += l_correct
                        val_total_nonpadding_yy += l_nonpadding

# auto encod    ing loss
                if opts.xx == 1:
                    with torch.cuda.device(0):
                        optimizer = loss_compute_gx.optimizer
                        loss_compute_gx.optimizer = None
                        decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gx_fx_x(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                        ntokens_src = (src_out != SRC.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                        loss_compute_gx.pad = SRC.vocab.stoi["<blank>"]
                        l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gx(decoder_output, src_out, ntokens_src, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                        loss_compute_gx.optimizer = optimizer
                        val_loss_all_xx += l
                        val_tokens_xx += ntokens_src
                        val_total_correct_xx += l_correct
                        val_total_nonpadding_xx += l_nonpadding

                if opts.yx == 1:
                    optimizer = loss_compute_gy.optimizer
                    loss_compute_gy.optimizer = None
                    with torch.cuda.device(0):
                        decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gy_inity_fx_x(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                        ntokens_trg = (trg_out != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                        loss_compute_gy.pad = TRG.vocab.stoi["<blank>"]
                        l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gy(decoder_output, trg_out, ntokens_trg, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                        loss_compute_gy.optimizer = optimizer
                        val_loss_all_yx += l
                        val_tokens_yx += ntokens_trg
                        val_total_correct_yx += l_correct
                        val_total_nonpadding_yx += l_nonpadding
                if opts.xy == 1:
                    with torch.cuda.device(0):
                        optimizer = loss_compute_gx.optimizer
                        loss_compute_gx.optimizer = None
                        decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions, log_self_attention_priors, self_attention_priors, log_enc_self_attention_priors, enc_self_attention_priors = model.forward_gx_initx_fy_y(src, trg, src_mask, trg_mask, src_in, src_out, trg_in, trg_out, src_mask_out, trg_mask_out, temperature, selftemperature=selftemperature, encselftemperature=encselftemperature)
                        ntokens_src = (src_out != SRC.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                        loss_compute_gx.pad = SRC.vocab.stoi["<blank>"]
                        l, l_xent, l_kl, l_correct, l_nonpadding = loss_compute_gx(decoder_output, src_out, ntokens_src, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions)
                        loss_compute_gx.optimizer = optimizer
                        val_loss_all_xy += l
                        val_tokens_xy += ntokens_src
                        val_total_correct_xy += l_correct
                        val_total_nonpadding_xy += l_nonpadding
            print("Val Result: PPL yx: %f, Acc yx: %f. PPL xy: %f, Acc xy: %f. PPL xx: %f, Acc xx: %f. PPL yy: %f, Acc yy: %f" %
                (math.exp(min(100, val_loss_all_yx / val_tokens_yx)), float(val_total_correct_yx)/val_total_nonpadding_yx, math.exp(min(100, val_loss_all_xy / val_tokens_xy)), float(val_total_correct_xy)/val_total_nonpadding_xy, math.exp(min(100, val_loss_all_xx / val_tokens_xx)), float(val_total_correct_xx)/val_total_nonpadding_xx, math.exp(min(100, val_loss_all_yy / val_tokens_yy)), float(val_total_correct_yy)/val_total_nonpadding_yy))
            torch.save({'model': model.state_dict(), 'src_vocab': SRC.vocab, 'trg_vocab': TRG.vocab, 'opts': opts, 'optimizer': optimizer}, '%s.e%d.pt'%(opts.save_to, epoch))
        model.train()

if __name__ == '__main__':
    main(opts)
