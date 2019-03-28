import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, mode, TRG, generator, criterion, devices, optimizer=None, model=None, chunk_size=5):
        # Send out to different gpus.
        self.mode = mode
        self.t_ = True
        self.pad = TRG.vocab.stoi["<blank>"]
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.devices = devices
        self.chunk_size = chunk_size
        self.count_ = 0
        
    def __call__(self, out, targets, normalize, log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions):
        total = 0.0
        total_xent = 0.0
        total_kl = 0.0
        total_nonpadding = 0
        total_correct = 0
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        generator = nn.parallel.replicate(self.generator, 
                devices=self.devices[:len(out_scatter)])
        criterion = nn.parallel.replicate(self.criterion, 
                devices=self.devices[:len(out_scatter)])
        out_grad = [[] for _ in out_scatter]
        flat_targets = targets.contiguous().view(-1)
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)


        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.optimizer is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            #num_correct = [item[0]
            #import pdb; pdb.set_trace()
            with torch.no_grad():
                num_correct = [((item[0].max(-1)[1].eq(item[1]))*(item[1].ne(self.pad))).float().sum().cpu().item() for item in y]
                num_correct = sum(num_correct)
                num_nonpadding = [(item[1].ne(self.pad)).float().sum().cpu().item() for item in y]
                num_nonpadding = sum(num_nonpadding)
                total_nonpadding += num_nonpadding
                total_correct += num_correct
            loss = nn.parallel.parallel_apply(criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum() / normalize
            total += l.item()
            total_xent += l.item()

            # Backprop loss to output of transformer
            if (self.optimizer is not None) and self.t_:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        total_kl = 0. 
        if self.mode == 'var' or self.mode == 'lstmvar':
            for log_prior_attention, prior_attention, log_posterior_attention, posterior_attention in zip(log_prior_attentions, prior_attentions, log_posterior_attentions, posterior_attentions):
                batch_size, h, trg_size, src_size = prior_attention.size()
                prior_attention = prior_attention.transpose(1, 2).contiguous().view(-1, h, src_size)
                log_prior_attention = log_prior_attention.transpose(1, 2).contiguous().view(-1, h, src_size)
                posterior_attention = posterior_attention.transpose(1, 2).contiguous().view(-1, h, src_size)
                log_posterior_attention = log_posterior_attention.transpose(1, 2).contiguous().view(-1, h, src_size)
                prior_attention = prior_attention[flat_targets.ne(self.pad)]
                log_prior_attention = log_prior_attention[flat_targets.ne(self.pad)]
                posterior_attention = posterior_attention[flat_targets.ne(self.pad)]
                log_posterior_attention = log_posterior_attention[flat_targets.ne(self.pad)]
                kl = (posterior_attention * (log_posterior_attention - log_prior_attention)).sum()
                #p = Categorical(prior_attention.view(-1, prior_attention.size(-1)))
                #q = Categorical(posterior_attention.view(-1, posterior_attention.size(-1)))
                #kl = kl_divergence(q, p).sum()
                total_kl += kl / normalize
            total += total_kl.item()

        # Backprop all loss through transformer.            
        if self.optimizer is not None:
            if self.mode == 'var' or self.mode == 'lstmvar':
                total_kl = total_kl
                total_kl.backward(retain_graph = True)
                total_kl = total_kl.item()
            if self.t_:
                out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
                o1 = out
                o2 = nn.parallel.gather(out_grad, 
                                        target_device=self.devices[0])
                o1.backward(gradient=o2, retain_graph=False)
            # nan-check
            nans = [
                (name, param)
                for name, param in self.model.named_parameters()
                if param.grad is not None and (param.grad != param.grad).any()
                ]
            if nans:
                print("FOUND NANS")
                #print([x[0] for x in nans])
                for _, param in nans:
                    param.grad[param.grad!=param.grad] = 0
            if self.count_ == self.accum_grad:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.count_ = 0
            self.count_ += 1
        return total * normalize, total_xent * normalize, total_kl * normalize, total_correct, total_nonpadding
