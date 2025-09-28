import torch

from logger import Logging, LogLevel

class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # `unbiased=False` uses n-1 for division, as opposed to n.
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

if __name__ == '__main__':
    torch.manual_seed(123)
    torch.set_printoptions(sci_mode=False)
    Logging.set_log_level(LogLevel.INFO)

    batch = torch.rand(2, 5) # 5 being the emb_dim

    # Left here for reference; mean_batch and var_batch are 0's and 1's as expected. The out_mean is 0's, but out_var
    # is not exactly 1's.
    #
    # mean = batch.mean(dim=-1, keepdim=True)
    # var = batch.var(dim=-1, keepdim=True)
    # norm_batch = (batch - mean) / torch.sqrt(var)

    # mean_batch = norm_batch.mean(dim=-1, keepdim=True)
    # var_batch = norm_batch.var(dim=-1, keepdim=True)

    # Logging.log(LogLevel.INFO, mean_batch)
    # Logging.log(LogLevel.INFO, var_batch)

    norm = LayerNorm(5)
    out = norm(batch)

    out_mean = out.mean(dim=-1, keepdim=True)
    out_var = out.var(dim=-1, keepdim=True, unbiased=False)

    Logging.log(LogLevel.INFO, f'out:\n{out}')
    Logging.log(LogLevel.INFO, f'out_mean:\n{out_mean}')
    Logging.log(LogLevel.INFO, f'out_var:\n{out_var}')
