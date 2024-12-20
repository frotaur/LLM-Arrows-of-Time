"""
Full definition of a GPT Language Model. 
Custom code (largely base on Andrej Karpathy's MinGPT)

TODO : Upgrade to flash attention (will be done soon) 
"""

import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchenhanced import DevModule
from torchenhanced import ConfigModule


class MinGPT(ConfigModule):
    """
    GPT Language Model.
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        embed_dim: int,
        n_heads: int,
        attn_length: int,
        mlp_ratio: float = 4,
        dropout: float = 0.1,
        embd_dropout: float = None,
        fast = True
    ):
        """
        Args:
            vocab_size: number of tokens in the vocabulary
            n_layer: number of transformer layers
            embed_dim: number of embedding dimensions
            n_heads: number of attention heads, must divie embed_dim
            attn_length: length of the attention window
            mlp_ratio: ratio of mlp hidden dim to embedding dim
            dropout: (optional) dropout probability
            embd_dropout: (optional) dropout probability for the embedding layer
            fast : Legacy parameter. Set to False to use old implementation, and allow
            loading of old models.
        """
        configo = dict(
            vocab_size=vocab_size,
            n_layers=n_layers,
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_length=attn_length,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            embd_dropout=embd_dropout,
        )
        super().__init__(configo)

        self.attn_length = attn_length

        if embd_dropout is None:
            embd_dropout = dropout

        block_config = dict(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_length=attn_length,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        if(fast):
            Block = FastBlock
        else:
            Block = SlowBlock
            
        self.transformer = nn.ModuleDict(
            dict(
                token_embedder=nn.Embedding(vocab_size, embed_dim),
                position_embedder=nn.Embedding(attn_length, embed_dim),
                drop=nn.Dropout(embd_dropout),
                h=nn.ModuleList([Block(**block_config) for _ in range(n_layers)]),
                ln_f=nn.LayerNorm(embed_dim),
            )
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        print("number of parameters: %.2fM" % (self.paranum / 1e6,))
        print(
            "Without head : %.2fM"
            % ((self.paranum - sum(p.numel() for p in self.lm_head.parameters())) / 1e6)
        )

    def forward(self, idx) -> torch.Tensor:
        """
        Process sequence of digits and outputs logits

        Args:
            idx: (B,T) sequence of TOKENIZED text.
                Max token integer must be <= self.vocab_size

        Returns:
            (B,T,vocab_size) Tensor of logits.
        """
        B, T = idx.shape
        assert (
            T <= self.attn_length
        ), f"Cannot forward sequence of length {T}, block size is only {self.attn_length}"

        # shape (1, T)
        pos = torch.arange(0, T, dtype=torch.long, device=self.device).unsqueeze(0)

        # forward the GPT model
        # token embeddings of shape (B, T, n_embd)
        tok_emb = self.transformer.token_embedder(idx)

        # position embeddings of shape (1, T, n_embd)
        pos_emb = self.transformer.position_embedder(pos)

        idx = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            idx = block(idx)
        idx = self.transformer.ln_f(idx)  # Still (B,T,n_embd)

        logits = self.lm_head(idx)  # (B,T,vocab_size) logits

        return logits  # (B,T,vocab_size)

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (B,T))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Use with model in inference mode (apply
        model.eval() first)

        Args:
            idx: (B,T) tensor of context tokens. Mostly, it will be B=1 but can
                do in parallel also
            max_new_tokens: number of tokens to generate on top of the
                conditioning sequence
            temperature: softmax temperature (lower -> more conservative
                sampling)
            do_sample: if True, use multinomial sampling. Otherwise use greedy
                decoding
            top_k: if set to int > 0, only sample from the top k most probable logits

        Returns:
            (B,T) LongTensor of generated token indices. Must still be decoded
            by tokenizer.
        """

        for _ in tqdm(range(max_new_tokens)):
            idx_next = self.generate_next_token(
                idx, temperature=temperature, do_sample=do_sample, top_k=top_k
            )

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_next_token(self, idx, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (B,T))
        and complete the sequence max_new_tokens times, feeding the predictions
        back into the model each time. Use with model in inference mode (apply
        model.eval() first)

        Args:
            idx: (B,T) tensor of context tokens. Mostly, it will be B=1 but can
                do in parallel also
            max_new_tokens: number of tokens to generate on top of the
                conditioning sequence
            temperature: softmax temperature (lower -> more conservative
                sampling)
            do_sample: if True, use multinomial sampling. Otherwise use greedy
                decoding
            top_k: if set to int > 0, only sample from the top k most probable
                logits

        Returns:
            next predicted token, Long
        """
        idx_cond = (
            idx if idx.shape[1] <= self.attn_length else idx[:, -self.attn_length :]
        )
        # forward the model to get the logits for the index in the sequence
        logits = self.forward(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        return idx_next


class CausalSelfAttention(DevModule):
    """
    Multi-head masked self-attention layer. Maybe in the future, benchmark
    against native pytorch implementation to make sure its not too slow/memory
    hungry

    Args:
        embed_dim: number of embedding dimensions
        n_heads: number of attention heads
        attn_length: length of the attention window
        dropout: (optional) dropout probability
    """

    def __init__(
        self, embed_dim: int, n_heads: int, attn_length: int, dropout: float = 0.1
    ):
        super().__init__()
        assert (
            embed_dim % n_heads == 0
        ), f"Number of heads {n_heads} must divide embedding dim {embed_dim}"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)

        # output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(attn_length, attn_length)).view(
                1, 1, attn_length, attn_length
            ),
        )
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=2)
        # (B, nh, T, hs)
        k = k.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.reshape(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        x = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        x = x.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        x = F.softmax(x, dim=-1)
        x = self.attn_dropout(x)

        x = x @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        x = x.transpose(1, 2).contiguous().reshape(B, T, C)

        # output projection
        x = self.resid_dropout(self.c_proj(x))

        return x

class FastCausalSelfAttention(DevModule):
    """
        Multi-head masked self-attention layer, implemented from pytorch,
        should be faster than the custom implementation.

        Args :
            embed_dim : number of embedding dimensions
            n_heads : number of attention heads
            attn_length : length of the attention window
            dropout : (optional) dropout probability 
    """

    def __init__(self, embed_dim : int, n_heads :int, attn_length:int, dropout:float = 0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, 'Number of heads {n_head} must divide embedding dim {n_embd}'
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.resid_dropout = nn.Dropout(dropout)

        self.attn_length = attn_length
        # Upper triangular part is 'TRUE', i.e., not allowed
        self.register_buffer("cant_attend", torch.tril(torch.ones((attn_length, attn_length),dtype=torch.int))==0)

    def forward(self, x):
        B, T, C = x.size()

        x, _ = self.attn(x, x, x, attn_mask=self.cant_attend[:T,:T], is_causal=True, need_weights=False) # (B, T, C)

        x = self.resid_dropout(x) # Apply the residual dropout

        return x

class FastBlock(DevModule):
    """
    One transformer block/layer, fast causal attention followed by a MLP.

    Args:
        embed_dim: number of embedding dimensions
        n_heads: number of attention heads
        attn_length: length of the attention window
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        dropout: (optional) dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_length: int,
        mlp_ratio: float,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = FastCausalSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_length=attn_length,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(embed_dim, int(mlp_ratio * embed_dim)),
                act=nn.GELU(),
                c_proj=nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
                dropout=nn.Dropout(dropout),
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp["dropout"](
            self.mlp["c_proj"](self.mlp["act"](self.mlp["c_fc"](self.ln_2(x))))
        )

        return x

class SlowBlock(DevModule):
    """
    One transformer block/layer, causal attention followed by a MLP.

    Args:
        embed_dim: number of embedding dimensions
        n_heads: number of attention heads
        attn_length: length of the attention window
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        dropout: (optional) dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        attn_length: int,
        mlp_ratio: float,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim=embed_dim,
            n_heads=n_heads,
            attn_length=attn_length,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(embed_dim, int(mlp_ratio * embed_dim)),
                act=nn.GELU(),
                c_proj=nn.Linear(int(mlp_ratio * embed_dim), embed_dim),
                dropout=nn.Dropout(dropout),
            )
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp["dropout"](
            self.mlp["c_proj"](self.mlp["act"](self.mlp["c_fc"](self.ln_2(x))))
        )

        return x
