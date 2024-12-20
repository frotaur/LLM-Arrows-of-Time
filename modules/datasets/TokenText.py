from torch.utils.data import Dataset
import torch
import os
import h5py

class TokenTextBOS(Dataset):
    """
    Dataset used to store tokenized text. Produces tuples of text, and the text shifted by one
    token, to be used as input and target for language modelling. Uses memory mapping, with hdf5.
    Adds a BOS token at the beginning, such that we don't 'miss' any token. As such, the 'effective'
    attention length is one less, as one token is dedicated to the <BOS> token.

    Args:
    text_location : location of the tokenized text tensor
    attn_length : size of the attention window TO BE CHANGED; IT IS ACTUALLY THE NUMBER OF TOKENS, NOT INCLUDE BOS
    stride : by how many tokens to stride to get the next example. Default is half the attention length.
    vocab_size : vocabulary_size such that we can add a BOS token at the beginning which will be vocab_size+1
    backwards : whether to read the text backwards. Default is False.
    permutation : list or tensor of ints between 0 and attn_length-1. If given, the text tensor will be permuted 
        according to the permutation, before appending the BOS token. If set, will ignore the 'backwards' argument.
    """

    def __init__(
        self, h5_file: str, attn_length: int, stride: int = None, backwards=False, permutation=None
    ):

        if(permutation is not None):
            assert len(permutation) == attn_length, "Permutation must have the same length as attn_length"
            if(isinstance(permutation, torch.Tensor)):
                assert set(permutation.tolist()) == set(range(attn_length)), "Permutation must be a permutation of the set of integers from 0 to attn_length-1"
            else:
                assert set(permutation) == set(range(attn_length)), "Permutation must be a permutation of the set of integers from 0 to attn_length-1"
        self.h5_file = h5_file
        self.attn_length = attn_length - 1  # -1 because we add a BOS token
        # attn_length is actually the length of the text that is produced. Adding the BOS, we get sentences of length attn_length+1
        self.backwards = backwards

        if stride is None:
            self.stride = self.attn_length // 2
        else:
            self.stride = stride

        if not os.path.isfile(self.h5_file):
            raise ValueError(f"File/Folder {self.h5_file} not found")

        self.h5_file = h5py.File(self.h5_file, "r")
        self.text_tensor = self.h5_file["tokens"]

        self.num_tokens = len(self.text_tensor)
        self.length = (self.num_tokens - self.attn_length - 1) // (
            self.stride
        )  # -1 because we need to have a target for each input
        
        print(
            f"Dataset contains {self.num_tokens/1e6:.2f}M tokens, resulting in {self.length//1000}k examples."
        )

        if(permutation is not None):
            self.permutation = torch.tensor(permutation,dtype=torch.long)
        elif(self.backwards):
            self.permutation = torch.arange(self.attn_length, -1, -1, dtype=torch.long)
        else:
            self.permutation = torch.arange(self.attn_length+1, dtype=torch.long)

    def __len__(self):
        return self.length

    def oldgetitem(self, idx):
        """
        Returns a tuple of (input, target) tensors, each of shape (attn_length)

        For now, when backwards, we still give the examples in the 'forward' way, but
        we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """
        true_idx = self.stride * (idx)
        
        if self.backwards:
            return self.add_BOS(
                torch.tensor(
                    self.text_tensor[true_idx + 1 : true_idx + self.attn_length + 1],
                    dtype=torch.long,
                ).flip(dims=(0,))
            ), torch.tensor(
                self.text_tensor[true_idx : true_idx + self.attn_length + 1],
                dtype=torch.long,
            ).flip(dims=(0,))
        else:
            return self.add_BOS(
                torch.tensor(
                    self.text_tensor[true_idx : true_idx + self.attn_length],
                    dtype=torch.long,
                )
            ), torch.tensor(
                self.text_tensor[true_idx : true_idx + self.attn_length + 1],
                dtype=torch.long,
            )


    def __getitem__(self, idx):
        """
        Returns a tuple of (input, target) tensors, each of shape (attn_length)

        For now, when backwards, we still give the examples in the 'forward' way, but
        we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """
        true_idx = self.stride * (idx)

        # all the indices that will be treated, including the target
        selected = self.text_tensor[true_idx : true_idx + self.attn_length + 1]


        return self.add_BOS(
                torch.tensor(
                    selected[self.permutation[:-1]],
                    dtype=torch.long,
                )), torch.tensor(
                    selected[self.permutation[:]],
                    dtype=torch.long,
                )
    
    def add_BOS(self, tens):
        """
        Adds a BOS token at the beginning of the tensor, and returns it.

        Args:
        tens : tensor of shape (attn_length)
        """
        return torch.cat(
            [torch.tensor([0], dtype=torch.long), tens], dim=0
        )  # (attn_length+1)

class TokenTextFWBW(TokenTextBOS):
    """
        Dataset used to store tokenized text.
        Returns sample randomly which are backward or forward,
        with different BOS tokens according to the direction.
        Allows to train a model in both directions at the same time.
    """
    def __init__(
        self, h5_file: str, attn_length: int, fw_token_num, bw_token_num, stride: int = None, 
    ):
        """
        Args:
        h5_file : location of the tokenized text tensor
        attn_length : size of the attention window TO BE CHANGED; IT IS ACTUALLY THE NUMBER OF TOKENS, NOT INCLUDE BOS
        fw_token_num : token number for forward direction (<|forward|>)
        bw_token_num : token number for backward direction (<|backward|>)
        stride : by how many tokens to stride to get the next example. Default is half the attention
        """
    
        super().__init__(h5_file, attn_length, stride, backwards=False)

        self.fw_token = fw_token_num
        self.bw_token = bw_token_num

        self.length = self.length*2 # double the length, as we will return both forward and backward examples

    def __getitem__(self, idx):
        """
        Returns a tuple of (input, target) tensors, each of shape (attn_length)

        For now, when backwards, we still give the examples in the 'forward' way, but
        we flip them. Maybe there is some reason why this is no bueno, but I don't think so.
        """
        true_idx = self.stride * (idx//2) # alternate between forward and backward

        do_flip = idx % 2 == 1 # alternate between forward and backward
        
        # all the indices that will be treated, including the target
        selected = torch.tensor(self.text_tensor[true_idx : true_idx + self.attn_length + 1], dtype=torch.long)

        if do_flip:
            selected = torch.flip(selected, dims=(0,))

        return self.add_BOS(
                    selected[:-1], backwards=do_flip), selected

    def add_BOS(self, tens, backwards=False):
        """
        Adds a BOS token at the beginning of the tensor, and returns it.

        Args:
        tens : tensor of shape (attn_length)
        """
        if(backwards):
            tok = self.bw_token
        else:
            tok = self.fw_token

        return torch.cat(
            [torch.tensor([tok], dtype=torch.long), tens], dim=0
        )  # (attn_length+1)