import torch

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, lr_begin=3e-4, lr_end=1e-5, anneal_steps=250000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.lr_begin = lr_begin
        self.lr_end = lr_end
        self.anneal_steps = anneal_steps
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        #rate = 1e-4
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        #return self.factor * \
        #    (self.model_size ** (-0.5) *
        #    min(step ** (-0.5), step * self.warmup ** (-1.5)))
        lr = self.lr_begin
        lr_end = self.lr_end
        anneal_steps = self.anneal_steps
        return max( 0, (lr - lr_end) * (anneal_steps - step) / anneal_steps ) + lr_end

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(model, WARMUP, d_model, lr_begin, lr_end, anneal_steps):
    return NoamOpt(d_model, 0.5, WARMUP,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), lr_begin=lr_begin, lr_end=lr_end, anneal_steps=anneal_steps)
def get_std_opt2(energy_network, model, WARMUP, d_model, lr_begin, lr_end, anneal_steps):
    return NoamOpt(d_model, 0.5, WARMUP,
            torch.optim.Adam(energy_network.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9), lr_begin=lr_begin, lr_end=lr_end, anneal_steps=anneal_steps)
