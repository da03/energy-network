import argparse
import sys
import math
import codecs

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
parser.add_argument("--output", required=True, help="Output path")
parser.add_argument("--direction", default="ende",
                    choices=["ende", "deen"],  help="Direction of translation")
parser.add_argument("--mode", default="soft",
                    choices=["soft", "var"],  help="Training model. Default to be vanilla transformer with soft attention.")
parser.add_argument("--beam_size", type=int, default=1, help="Beam Size.")
parser.add_argument("--max_trg_len", type=int, default=150, help="Maximum Trg Length")
parser.add_argument("--model", default="", help="Model Path")
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

    train, val = torchtext.datasets.TranslationDataset.splits(
            path=opts.dataset, train='train/train.tags.en-de.bpe',
            validation='dev/valid.en-de.bpe', test=None,
            exts=exts,
            fields=[('src', SRC), ('trg', TRG)])

    BATCH_SIZE = 1
    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, train=False, shuffle=False, sort=False)
    # Build Model
    assert opts.model != ''
    print ('Loading Model from %s'%opts.model)
    checkpoint = torch.load(opts.model)
    SRC.vocab = checkpoint['src_vocab']
    TRG.vocab = checkpoint['trg_vocab']
    src_vocab_size = len(SRC.vocab.itos)
    trg_vocab_size = len(TRG.vocab.itos)
    sharelstm = checkpoint['opts'].sharelstm if hasattr(checkpoint['opts'], 'lstm') else 0
    residual_var = checkpoint['opts'].residualvar if hasattr(checkpoint['opts'], 'residualvar') else 0
    print ('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))
    #model = make_model(checkpoint['opts'].mode, src_vocab_size, trg_vocab_size, n_enc=5, n_dec=5,
    #               d_model=278, d_ff=507, h=2, dropout=0.1)
    model = make_model(checkpoint['opts'].mode, src_vocab_size, trg_vocab_size, n_enc=5, n_dec=5,
                   d_model=278, d_ff=507, h=2, dropout=0.1, share_decoder_embeddings=checkpoint['opts'].share_decoder_embeddings,
                   share_word_embeddings=checkpoint['opts'].share_word_embeddings, dependent_posterior=checkpoint['opts'].dependent_posterior,
                   sharelstm=sharelstm, residual_var=residual_var)
    model.load_state_dict(checkpoint['model'])
    model.trg_pad = TRG.vocab.stoi["<blank>"]
    model.cuda()
    optimizer = checkpoint['optimizer']

    weight = torch.ones(trg_vocab_size)
    weight[TRG.vocab.stoi["<blank>"]] = 0
    criterion = torch.nn.NLLLoss(weight, size_average=False)
    criterion.cuda()
                                   
    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0
    start = time.time()
    model.eval()

    BOS = TRG.vocab.stoi[BOS_WORD]
    EOS = TRG.vocab.stoi[EOS_WORD]
    model.decoder.vocab = TRG.vocab
    with torch.no_grad():
        with codecs.open(opts.output, 'w', 'utf-8') as fout:
            for i, batch in enumerate(val_iter):
                src = batch.src[0].transpose(0, 1) # 1, len
                trg = batch.trg.transpose(0, 1) # 1, len

                assert src.size(0) == 1
                src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)

                trg = batch.trg.transpose(0, 1) # batch, len
                trg_mask = (trg != TRG.vocab.stoi["<blank>"]).unsqueeze(-2)
                trg_mask = trg_mask & Variable(subsequent_mask(trg.size(-1)).to(trg_mask))
                trg_y = trg[:, 1:]
                ntokens = (trg_y != TRG.vocab.stoi["<blank>"]).data.view(-1).sum().item()
                #decoder_output, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions = model(src, trg, src_mask, trg_mask, None)
                #word_probs = model.generator(decoder_output)
                #model.decoder.decoder_output = decoder_output
                model.decoder.trg_mask = trg_mask
                model.decoder.src_mask = src_mask

                if checkpoint['opts'].encselfmode != 'soft':
                    encselftemperature = -1
                    encselfdependent_posterior = 1
                    encattn_dropout = False
                else:
                    encselftemperature = None
                    encselfdependent_posterior = 0
                    encattn_dropout = True

                if checkpoint['opts'].mode == 'soft': # vanilla beam search
                    if checkpoint['opts'].selfmode == 'soft':
                        hypothesis, score = model.beam_search_soft(opts.beam_size, src, src_mask, BOS, EOS, opts.max_trg_len, encselftemperature=encselftemperature, encselfdependent_posterior=encselfdependent_posterior, encattn_dropout=encattn_dropout)
                    else:
                        hypothesis, score = model.beam_search_soft_selfhard(opts.beam_size, src, src_mask, BOS, EOS, opts.max_trg_len, encselftemperature=encselftemperature,encselfdependent_posterior=encselfdependent_posterior,encattn_dropout=encattn_dropout)
                else:
                    if not hasattr(checkpoint['opts'], 'selfmode') or checkpoint['opts'].selfmode == 'soft':
                        hypothesis, score = model.beam_search_hard(opts.beam_size, src, src_mask, BOS, EOS, opts.max_trg_len, encselftemperature=encselftemperature, encselfdependent_posterior=encselfdependent_posterior, encattn_dropout=encattn_dropout)
                    else:
                        hypothesis, score = model.beam_search_hard_selfhard(opts.beam_size, src, src_mask, BOS, EOS, opts.max_trg_len, encselftemperature=encselftemperature, encselfdependent_posterior=encselfdependent_posterior, encattn_dropout=encattn_dropout)
                words = []
                for word_id in src.view(-1):
                    words.append(SRC.vocab.itos[word_id.item()])
                try:
                    print ('Src: %s'%(' '.join(words)))
                except Exception as e:
                    pass
                words = []
                for word_id in trg.view(-1)[1:-1]:
                    words.append(TRG.vocab.itos[word_id.item()])
                try:
                    print ('Ground Truth: %s'%(' '.join(words)))
                except Exception as e:
                    pass
                words = []
                for word_id in hypothesis:
                    if word_id.item() == EOS:
                        break
                    words.append(TRG.vocab.itos[word_id.item()])
                try:
                    print ('Predicted (%f): %s'%(score, ' '.join(words)))
                except Exception as e:
                    pass
                fout.write(' '.join(words)+'\n')


if __name__ == '__main__':
    main(opts)
