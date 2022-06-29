import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PositionalEncoding:
    def __init__(self, L: int):
        input_dims = 3
        embed_fns = []
        # include input f(x) = x
        embed_fns.append(lambda x: x)
        out_dim = input_dims

        max_freq = L-1
        N_freqs = L

        freq_bands = 2.**torch.linspace(0., max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_positional_encoder(L: int):
    embedder_obj = PositionalEncoding(L)
    def pos_encoder(x, eo=embedder_obj): return eo.embed(x)
    return pos_encoder, embedder_obj.out_dim


if __name__ == '__main__':
    pos_encoder, out_dim = get_positional_encoder(10)
    print(pos_encoder)
    print(out_dim)
    pos_encoder, out_dim = get_positional_encoder(4)
    print(pos_encoder)
    print(out_dim)
