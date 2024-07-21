"""
Full definition of a GPT Language Model.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchenhanced import ConfigModule


class MinGRU(ConfigModule):
    """
    GRU for language modeling.

    Args:
        vocab_size: number of tokens in the vocabulary
        embed_dim: number of embedding dimensions
        desired_attn_length: length of the attention window that will be used
            in training. Has no effect on the model
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        dropout: (optional) dropout probability
        embd_dropout: (optional) dropout probability for the embedding layer
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        desired_attn_length: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        embd_dropout: float = None,
    ):
        configo = dict(
            vocab_size=vocab_size,
            n_layers=n_layers,
            embed_dim=embed_dim,
            desired_attn_length=desired_attn_length,
            dropout=dropout,
            embd_dropout=embd_dropout,
        )

        super().__init__(configo)

        if embd_dropout is None:
            embd_dropout = dropout

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.token_embedder = nn.Embedding(vocab_size, embed_dim)
        self.embd_dropout = nn.Dropout(embd_dropout)

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        print(f'{"gru :"}'.capitalize())
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
            (B,T,vocab_size) Tensor of logits
        """
        B, T = idx.shape
        # assert T <= self.attn_length, f"Cannot forward sequence of length {T}, block size is only {self.attn_length}"

        # forward the GPT model
        # token embeddings of shape (B, T, n_embd)
        idx = self.embd_dropout(self.token_embedder(idx))

        idx, hf = self.gru(idx)  # (B,T,n_embd) (use default zeros for h0)

        assert hf.shape == (
            self.config["n_layers"],
            B,
            self.config["embed_dim"],
        ), f"hf shape \
          is wrong : {hf.shape} ! it does batch_first=True, when it shouldn't !!"

        idx = self.ln_f(idx)  # Still (B,T,n_embd)

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
            top_k: if set to int > 0, only sample from the top k most probable
                    logits

        Returns:
            (B,T) LongTensor of generated token indices. Must still be decoded by tokenizer.
        """
        B, T = idx.shape
        last_tok = (idx[:, -1])[:, None]  # (B,1)

        # First, run the context through the GRU :
        context = self.token_embedder(idx[:, :-1])  # (B,T-1,n_embd)
        _, hf = self.gru(context)  # (n_layers,B,n_embd) (use default zeros for h0)

        # Then, generate, giving only last hf and last token as input, for speed :
        for _ in range(max_new_tokens):
            last_tok, hf = self.generate_next_token(
                last_tok, hf, temperature=temperature, do_sample=do_sample, top_k=top_k
            )

            idx = torch.cat((idx, last_tok), dim=1)

        return idx

    @torch.no_grad()
    def generate_next_token(
        self, idx, hidden, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a condition hidden state and and token index and generates the
        next token.

        Args:
            idx: (B,1) tensor of context token. Mostly, it will be B=1 but can
                do in parallel also
            hidden: (B,n_layers,n_embd) tensor of hidden states @@MAYBE BATCH
                FIRST DOES NOT WORK ON IT, CHECK IT@@
            temperature: softmax temperature (lower -> more conservative
                sampling)
            do_sample: if True, use multinomial sampling. Otherwise use greedy
                decoding
            top_k: if set to int > 0, only sample from the top k most probable
                logits

        Returns:
            (next predicted token, (B,1) longs), (last hidden state (n_layers,B,n_embd))
        """
        idx = self.token_embedder(idx)  # (B,1,n_embd)
        # forward the model to get the logits for the index in the sequence
        out, hf = self.gru(idx, hidden)  # (B,1,n_embd) (1,B,n_embd)

        logits = self.lm_head(self.ln_f(out))  # (B,1,vsize)

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
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)  # (B,1)

        return idx_next, hf


class MinLSTM(ConfigModule):
    """
    Simple LSTM for language modeling.

    Args:
        vocab_size: number of tokens in the vocabulary
        embed_dim: number of embedding dimensions
        desired_attn_length: length of the attention window that will be used
            in training. Has no effect on the model
        mlp_ratio: ratio of mlp hidden dim to embedding dim
        dropout: (optional) dropout probability
        embd_dropout: (optional) dropout probability for the embedding layer
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        desired_attn_length: int,
        n_layers: int = 1,
        dropout: float = 0.1,
        embd_dropout: float = None,
    ):
        configo = dict(
            vocab_size=vocab_size,
            n_layers=n_layers,
            embed_dim=embed_dim,
            desired_attn_length=desired_attn_length,
            dropout=dropout,
            embd_dropout=embd_dropout,
        )

        super().__init__(configo)

        if embd_dropout is None:
            embd_dropout = dropout

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.token_embedder = nn.Embedding(vocab_size, embed_dim)
        self.embd_dropout = nn.Dropout(embd_dropout)

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        print(f'{"lstm:"}'.capitalize())
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
            (B,T,vocab_size) Tensor of logits
        """
        B, T = idx.shape
        # assert T <= self.attn_length, f"Cannot forward sequence of length {T}, block size is only {self.attn_length}"

        # forward the GPT model
        idx = self.embd_dropout(
            self.token_embedder(idx)
        )  # token embeddings of shape (B, T, n_embd)

        idx, (hf, cf) = self.lstm(idx)  # (B,T,n_embd) (use default zeros for h0)

        assert (
            hf.shape
            == (
                self.config["n_layers"],
                B,
                self.config["embed_dim"],
            )
        ), f"hf shape  is wrong : {hf.shape} ! it does batch_first=True, when it shouldn't !!"

        idx = self.ln_f(idx)  # Still (B,T,n_embd)

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
            top_k: if set to int > 0, only sample from the top k most probable
                logits

        Returns:
            (B,T) LongTensor of generated token indices. Must still be decoded by tokenizer.
        """
        B, T = idx.shape
        last_tok = (idx[:, -1])[:, None]  # (B,1)

        # First, run the context through the GRU :
        context = self.token_embedder(idx[:, :-1])  # (B,T-1,n_embd)
        # (n_layers,B,n_embd) (use default zeros for h0)
        _, (hf, cf) = self.lstm(context)

        # Then, generate, giving only last hf and last token as input, for speed :
        for _ in range(max_new_tokens):
            last_tok, (hf, cf) = self.generate_next_token(
                last_tok,
                (hf, cf),
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
            )

            idx = torch.cat((idx, last_tok), dim=1)

        return idx

    @torch.no_grad()
    def generate_next_token(
        self, idx, hidden, temperature=1.0, do_sample=False, top_k=None
    ):
        """
        Take a condition hidden state and and token index and generates the
        next token.

        Args:
            idx: (B,1) tensor of context token. Mostly, it will be B=1 but can
                do in parallel also
            hidden: 2-uple of (n_layers,B,n_embd) tensor of hidden states
            temperature: softmax temperature (lower -> more conservative
                sampling)
            do_sample: if True, use multinomial sampling. Otherwise use greedy
                decoding
            top_k: if set to int > 0, only sample from the top k most probable
                logits

        Returns:
            (next predicted token, (B,1) longs), (last hidden state (n_layers,B,n_embd))
        """
        idx = self.token_embedder(idx)  # (B,1,n_embd)
        # forward the model to get the logits for the index in the sequence
        out, (hf, cf) = self.lstm(idx, hidden)  # (B,1,n_embd) (1,B,n_embd)

        logits = self.lm_head(self.ln_f(out))  # (B,1,vsize)

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
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)  # (B,1)

        return idx_next, (hf, cf)
