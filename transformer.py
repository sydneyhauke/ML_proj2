from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

# ------------------------- #
# Reimplementation of GPT-2 #
# ------------------------- #

class AttentionHead(torch.nn.Module):

    def __init__(self, embedding_size: int, out_size: int, layer:int=None, n:int=None) -> None:
        """A single attention head.

        Args:
            embedding_size (int): The size of the input embedding.
            out_size (int): The size of the output embedding.
            layer (int): The layer number, for debugging purposes.
            n (int): The head number, for debugging purposes.
        """
        super().__init__()

        self.layer = layer
        self.n = n

        self.queries = torch.nn.Parameter(torch.randn(embedding_size, out_size))
        self.keys = torch.nn.Parameter(torch.randn(embedding_size, out_size))
        self.values = torch.nn.Parameter(torch.randn(embedding_size, out_size))

        # initialize the parameters
        torch.nn.init.kaiming_normal_(self.queries)
        torch.nn.init.kaiming_normal_(self.keys)
        torch.nn.init.kaiming_normal_(self.values)

    @typechecked
    def forward(
        self, x: TensorType['batch', 'tokens', 'embedding_size']
    ) -> TensorType['batch', 'tokens', 'out_size']:

        q = torch.einsum("bte,ei->bti", x, self.queries)
        k = torch.einsum("bte,ei->bti", x, self.keys)
        v = torch.einsum("bte,ei->bti", x, self.values)

        debug(q, self.layer, self.n, 'q')
        debug(k, self.layer, self.n, 'k')
        debug(v, self.layer, self.n, 'v')

        qk = torch.einsum("bti,bTi->btT", q, k)
        s = torch.softmax(qk / q.shape[-1]**0.5, dim=-1)
        out = torch.einsum("btT,bTi->bti", s, v)

        debug(qk, self.layer, self.n, 'fit')
        debug(s, self.layer, self.n, 'attention')
        debug(out, self.layer, self.n, 'head')

        return out

class MultiAttentionHead(torch.nn.Module):

    def __init__(self, embedding_size: int, heads: int, layer:int=None) -> None:
        assert embedding_size % heads == 0, "embedding size must be divisible by heads."
        super().__init__()
        self.layer = layer

        out_size = embedding_size // heads

        self.heads = torch.nn.ModuleList(
            [AttentionHead(embedding_size, out_size, layer=layer, n=n) for n in range(heads)])
        self.weight = torch.nn.Parameter(torch.randn(embedding_size, embedding_size))

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x: TensorType['b', 't',
                                    'emb']) -> TensorType['b', 't', 'out']:
        combined = torch.cat([head(x) for head in self.heads], dim=-1)
        debug(combined, self.layer, "heads-combined")
        multihead = combined @ self.weight
        debug(multihead, self.layer, "multihead")
        output = multihead + x
        debug(output, self.layer, "layer")
        return output


class ResidualMLP(torch.nn.Module):
    def __init__(self, embeding_size: int, *layers_dims: int, layer:int=None) -> None:
        """An MLP with residual a connection.

        Args:
            embedding_size (int): The size of the input and output embedding.
            *layers_dims (int): The dimensions of the hidden layers.
            layer (int): The layer number, for debugging purposes.
        """

        super().__init__()
        self.layer = layer
        dims = embeding_size, *layers_dims, embeding_size
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        # HeWeight initialization
        for layer in self.layers:
            torch.nn.init.kaiming_normal_(layer.weight)


    def forward(self, x: TensorType['batch', 'tokens', 'embedding_size']) -> TensorType['batch', 'tokens', 'embedding_size']:
        initial = x
        for l, layer in enumerate(self.layers):
            x = layer(x)
            debug(x, self.layer, "mlp", l)
            x = torch.relu(x)
            debug(x, self.layer, "mlp", l, "relu")

        x = x + initial
        debug(x, self.layer, "mlp", "residual")
        return x

class Transformer(torch.nn.Module):

    def __init__(self, voc_size: int, embedding_size: int, depth: int,
                 heads: int, # pos_encoder: TensorType['max_prompt_len', 'embedding_size'],
                 mlp_dims: Optional[Tuple[int, ...]]=None) -> None:
        super().__init__()

        self.depth = depth
        self.heads = heads
        self.mlp_dims = mlp_dims
        self.voc_size = voc_size
        self.embedding = torch.nn.Embedding(voc_size, embedding_size)
        # self.position_encoder = PositionEncoder()
        self.position_encoder = torch.nn.Parameter(torch.randn(2, embedding_size))

        heads = [
            MultiAttentionHead(embedding_size, heads, layer=layer)
            for layer in range(depth)
        ]
        if mlp_dims is not None:
            mlps = [ResidualMLP(embedding_size, *mlp_dims, layer=layer)
                for layer in range(depth)]
            # Interleave heads and mlps
            blocks = [block for layer in zip(heads, mlps) for block in layer]
        else:
            blocks = heads

        self.blocks = torch.nn.Sequential(*blocks)
        self.unembedding = torch.nn.Parameter(torch.rand(embedding_size, voc_size))

        # initialisation
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        torch.nn.init.xavier_uniform_(self.position_encoder)
        torch.nn.init.xavier_uniform_(self.unembedding)

    def forward(self, x: TensorType['batch', 'token']) -> List[str]:
        embeded = self.embedding(x)
        debug(embeded, "embeded")
        # with_pos = self.position_encoder(embeded)
        with_pos = embeded + self.position_encoder
        debug(with_pos, "embed+pos")
        x = self.blocks(with_pos)
        out = x[:, -1, :].squeeze(1)  # only the last token is the prediction
        unembeded = out @ self.unembedding
        debug(unembeded, "unembeded")
        # probas = torch.softmax(unembeded, dim=-1)
        # debug(probas, "probas")
        # return probas
        return unembeded


# ------------------------------ #
# Help for visualizing the model #
# ------------------------------ #

DEBUG = set()  # debug nothing
EllipsisType = type(...)
DEBUG_CALLBACK = None

def debug(value: TensorType, *name: Union[str, int]) -> None:
    for pattern in DEBUG:
        # print('eval', pattern, name)
        for part, pat in zip(name, pattern):
            if pat is not ... and part != pat:
                break  # not a match, next pattern
        else:
            # We have found a pattern that matches
            if DEBUG_CALLBACK is not None:
                DEBUG_CALLBACK(value, *name)
            else:
                print(*name)
                pprint_2d_tensor(value)
            return


def set_debug(*args: Union[str, List[Union[str, int, EllipsisType]]], callback=None) -> None:
    """Print matrices whose name correspond to the pattern

    Examples:
        set_debug() will print nothing.
        set_debug(()) will print everthing possible.
        set_debug(1) will print everything happening the first layer.
        set_debug([1, 2, "head"]) will print the output of the second head of the first layer.
        set_debug([1, ..., "q"]) will print all queries of the first layer.
        set_debug([..., ..., "q"], [..., ..., "k"]) will print all queries and keys of all layers.

    Optionnally, a callback can be provided to perfom an action instead of printing.
    """

    args = [a if isinstance(a, tuple) else (a,) for a in args]
    DEBUG.clear()
    DEBUG.update(args)
    global DEBUG_CALLBACK
    DEBUG_CALLBACK = callback

@contextmanager
def temp_debug(*args: Union[str, List[Union[str, int, EllipsisType]]], callback) -> None:
    """Same as set_debug but only for the duration of the context."""
    old_debug = DEBUG.copy()
    old_callback = DEBUG_CALLBACK
    set_debug(*args, callback=callback)
    try:
        yield
    finally:
        set_debug(*old_debug, callback=old_callback)
