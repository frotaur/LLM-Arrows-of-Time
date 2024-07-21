# -*- coding: utf-8 -*-

import os
import json
import pathlib
import argparse
from tqdm import tqdm

import torch

from transformers import logging
from transformers import AutoTokenizer

logging.set_verbosity_error()  # needed to stop the stupid warning messages

##################
### MINI-UTILS ###
##################


def log(*args):
    print(*args)


def all_none(*args):
    return all([arg is None for arg in args])


def all_not_none(*args):
    return all([arg is not None for arg in args])


def is_any_none(*args):
    return not all_not_none(*args)


def any_not_none(*args):
    return any([arg is not None for arg in args])


def first_not_none(*args):
    return (
        None
        if len(args) == 0
        else args[0]
        if args[0] is not None
        else first_not_none(*args[1:])
    )


def multiget(d, vals):
    return [d.get(val) for val in vals]


def remove_none_vals(d):
    return {k: d[k] for k in d if d[k] is not None}


def put_and_return(d, k, v):
    return d.update({k: v}) or v


def is_file(file_path):
    return os.path.isfile(file_path)


def are_files(*file_paths):
    return all([is_file(file_path) for file_path in file_paths])


def list_dir_files(dir_path):  # returns the files in a directory
    return [
        file_name
        for file_name in os.listdir(dir_path)
        if is_file(os.path.join(dir_path, file_name))
    ]


def load_string(file_path):
    try:
        return pathlib.Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        print(f"Load_string encountered an error when trying to load {file_path}")
        print(f"Exception {e=}")
        return ""


def save_string(string, file_path):
    try:
        pathlib.Path(file_path).write_text(string, encoding="utf-8")
    except Exception as e:
        return log(f"Error writing to {file_path=}, {e}")


def load_json(file_path):
    return json.loads(load_string(file_path))


def save_json(obj, file_path):
    save_string(json.dumps(obj), file_path) or obj


def load_torch(file_path, *, device=None, **params):
    return torch.load(
        file_path, map_location=(device if device is not None else "cpu"), **params
    )  # wrapper to unify notation


def save_torch(obj, file_path, *, replace=True, **params):
    if not is_file(file_path) or replace:
        return torch.save(obj, file_path, **params) or obj  # wrapper to unify notation
    else:
        return obj


def sanitize_quotes(text):
    return text.replace("’", "'").replace("”", '"').replace("“", '"')


def get_first_directory(path):
    return next((entry.name for entry in os.scandir(path) if entry.is_dir()), None)


####################
### TOKENIZATION ###
####################


class Tokenizer:
    """The base class for the tokenizers"""

    def __init__(self):
        self.vs = 1

    def tokenize(self, s, **params):
        pass  # tokenizes a string into a batched tensor [1, t], t = num tokens of s

    def detokenize(self, tokens, *, join=True):
        """
        Converts a pytorch tensor of tokens into a string or a list of strings
        Args:
            tokens: size [t] or size [b, t] (in the latter case, will be converted to [b * t])
            join: whether to put the tokens together in a string or leave them as a list of words
        """
        pass  # detokenizes a pytorch tensor into a string

    def tokenize_txt_to_pt(self, txt, *, add_if_unknown=False, dtype=None):
        tokens = self.tokenize(txt, add_if_unknown=add_if_unknown)
        if dtype is not None and isinstance(dtype, torch.dtype):
            tokens = tokens.to(dtype)
        log(
            f"Converted string length {len(txt)} into tensor of shape {tokens.shape} and of type {tokens.dtype}"
        )
        return tokens

    def tokenize_txt_to_pt_chunks(self, txt, *, add_if_unknown, dtype=None):
        # Splitting the text into chunks
        chunk_size = 1 * 1024 * 1024  # 10M tokens chunk size
        chunks = [txt[i : i + chunk_size] for i in range(0, len(txt), chunk_size)]

        tokenized_chunks = []
        print("Tokenizing... ")
        for chunk in tqdm(chunks):
            tokens = self.tokenize(chunk, add_if_unknown=add_if_unknown)
            if dtype is not None and isinstance(dtype, torch.dtype):
                tokens = tokens.to(dtype)

            tokenized_chunks.append(tokens)

        print(f"Chunk size : {tokenized_chunks[0].shape}*{(1,len(chunks))}")
        # If you want to return all chunks together:
        return torch.cat(tokenized_chunks, dim=1)

    def tokenize_txt_to_pt_file(
        self, txt, pt_file_path, *, add_if_unknown=False, dtype=None
    ):
        """Converts a string to a torch.Long tensor made of the tokens, which we save into the pf_file_name"""
        save_torch(
            self.tokenize_txt_to_pt_chunks(
                txt, add_if_unknown=add_if_unknown, dtype=dtype
            ),
            pt_file_path,
        )

    def tokenize_txt_file_to_pt_file(
        self, txt_file_path, pt_file_path, *, add_if_unknown=False, dtype=None
    ):
        """Converts a text file to a torch.Long tensor made of the tokens, which we save into the pf_file_name"""
        self.tokenize_txt_to_pt_file(
            load_string(txt_file_path),
            pt_file_path,
            add_if_unknown=add_if_unknown,
            dtype=dtype,
        )


class MinTokenizer(Tokenizer):
    """A basic tokenizer class that assumes that all tokens are one-character long, and that learns them on the fly"""

    def __init__(self, **params):
        super().__init__()
        # A bit hacky, but we should have a vs size consistent with the number of tokens
        (self.t2i, self.i2t, self.vs) = (
            {"[?]": 0},
            {0: "[?]"},
            1,
        )  # token to int and int to token, vocab size
        if (file_name := params.get("file_name")) is not None:
            self.load_from_json(file_name)

    def add_token(self, t, i=None):
        if t in self.t2i:
            return
        if i is None:
            i = len(self.t2i)
        (self.t2i[t], self.i2t[i]) = (i, t)
        self.vs = len(self.t2i)  # equal to len(self.i2t) as well
        return i

    def get_token_i(self, t, *, add_if_unknown=False):
        return self.add_token(t) if add_if_unknown else self.t2i.get(t, 0)

    def build_tokens_from_string(self, string):
        for t in string:
            self.add_token(t)

    def tokenize(self, s, *, add_if_unknown=False):
        token_ids = [
            self.get_token_i(token, add_if_unknown=add_if_unknown) for token in s
        ]  # list of t tokens
        return torch.tensor(token_ids)[None, :]  # [1, t]

    def detokenize(self, tokens, *, join=True):
        if len(tokens.shape) > 1:
            tokens = tokens.view(-1)  # [b, t] -> [b * t]
        token_chars = [self.i2t.get(i, f"[{i}]?") for i in tokens.tolist()]
        return "".join(token_chars) if join else token_chars

    def save_to_json(self, file_name):
        save_json(dict(t2i=self.t2i), file_name)

    def load_from_json(self, file_name):
        if "t2i" in (obj := load_json(file_name)):
            for k, v in obj["t2i"].items():
                self.add_token(k, int(v))
        else:
            log(f"Could not load from {file_name=}: invalid {obj=}")


class HfTokenizer(Tokenizer):
    """A base class for the HF tokenizers (gpt2, autotokenizer)"""

    def __init__(
        self, *, m_name=None, m_path=None, st_index=None, et_index=None
    ):  # start token index, end token index
        (self.hf_tokenizer, self.m_name) = (
            load_pretrained_hf_tokenizer(m_name=m_name, m_path=m_path),
            m_name,
        )
        self.vs = self.hf_tokenizer.vocab_size  #
        if self.m_name is None:
            self.m_name = ""

        (self.st_index, self.et_index) = (
            None,
            None,
        )  # start and end token indices (may need to be hacked)

    def tokenize(self, s, **params):
        if self.m_name.startswith("gpt2"):
            s = sanitize_quotes(s)  # {?} a bit hacky
        return self.hf_tokenizer(
            s, return_tensors="pt", verbose=True
        ).input_ids  # [1, t]

    def detokenize(self, tokens, *, join=True):
        tokens = tokens.to(dtype=torch.int64)  # To convert other types in case
        if len(tokens.shape) > 1:
            tokens = tokens.view(-1)  # [b, t] -> [b * t]
        t_strings = [self.hf_tokenizer.decode(tokens)]

        return "".join(t_strings) if join else t_strings


def load_pretrained_hf_tokenizer(*, m_name: str = None, m_path: str = None):
    """
    Loads a pre-trained Hugging Face tokenizer
    Args:
        m_name: the name of the model (e.g. 'gpt2', 'gpt2-large', 'gpt2-xl', 'bert-base-uncased') to be downloaded if necessary
        m_path: the file path to the tokenizer (if stored in a directory)
    """
    if m_path is not None:
        return AutoTokenizer.from_pretrained(m_path, use_fast=True)
    elif m_name is not None:
        return AutoTokenizer.from_pretrained(m_name, use_fast=True)
    else:
        return None


tokenizers_by_name = {}

######################
### MAIN FUNCTIONS ###
######################


def get_tokenizer(
    *, m_name: str = None, tokenizer_name: str = None, m_path: str = None
):
    """
    Args:
        tokenizer_name (optional): if there is a tokenizer with that name in
            tokenizers_by_name
        m_name (optional): a HuggingFace tokenizer name e.g. 'gpt2',
            'gpt2-large', 'facebook/opt-125m', or 'bert-base-uncased'
        m_path (optional): the path to the (HuggingFace) tokenizer (saved
            offline)
    """
    return (
        HfTokenizer(m_name=m_name, m_path=m_path)
        if any_not_none(m_name, m_path)
        else tokenizers_by_name.get(tokenizer_name)
    )


def tokenize_txt_file(
    txt_file_path: str,
    pt_file_path: str,
    *,
    tokenizer_name: str = None,
    m_name: str = None,
    m_path: str = None,
    dtype: torch.dtype = None,
):
    """
    Converts a txt file into a pytorch file containing the ids of the tensors
    (in format [1, t] where t is the number of tokesn)

    Args:
        txt_file_path: the location of the source text file
        pt_file_path: the location of the destination pytorch token file
        tokenizer_name (optional): if there is a tokenizer with that name in
            tokenizers_by_name
        m_name (optional): a HuggingFace tokenizer name e.g. 'gpt2',
            'gpt2-large', 'facebook/opt-125m', or 'bert-base-uncased'
        m_path (optional): the path to the (HuggingFace) tokenizer (saved
            offline)
        dtype (optional): by default, the token tensors are int64, but another
            type (such as int16) can be specified
    """
    global tokenizers_by_name
    tokenizer = get_tokenizer(
        tokenizer_name=tokenizer_name, m_name=m_name, m_path=m_path
    )
    if tokenizer is None:
        return log("Could not build tokenizer")
    tokenizer.tokenize_txt_file_to_pt_file(txt_file_path, pt_file_path, dtype=dtype)


def tokenize_dir_txt_files_from_tok(
    dir_path: str,
    pt_file_path: str,
    *,
    tokenizer: HfTokenizer = None,
    dtype: torch.dtype = None,
):
    if pt_file_path.endswith(".pt"):
        pt_file_path = pt_file_path[: -len(".pt")]
    if pt_file_path.endswith(".txt"):
        pt_file_path = pt_file_path[: -len(".txt")]

    txt_file_paths = []
    for subdir, _, files in os.walk(dir_path):
        txt_file_paths.extend(
            [os.path.join(subdir, file) for file in files if file.endswith(".txt")]
        )

    part_index = 0
    big_txt = ""  # A concatenation of the files

    def save_and_clear_big_txt(big_txt, part_index):
        part_txt_file_path = pt_file_path + f"_{part_index:04}_bigtext.txt"
        save_string(big_txt, part_txt_file_path)
        part_pt_file_path = pt_file_path + f"_{part_index:04}.pt"
        tokenizer.tokenize_txt_file_to_pt_file(
            part_txt_file_path, part_pt_file_path, dtype=dtype
        )
        os.remove(part_txt_file_path)
        log(f"Saved and cleared part {part_index:04}")

    print("Starting tokenization of files")
    for txt_file_path in tqdm(txt_file_paths):
        print(f"Adding {txt_file_path} to big_txt")
        txt = load_string(txt_file_path) + ("\n" * 5)
        big_txt += txt

    save_and_clear_big_txt(big_txt, part_index)


def tokenize_dir_txt_files(
    dir_path: str,
    pt_file_path: str,
    *,
    tokenizer_name: str = None,
    m_name: str = None,
    m_path: str = None,
    dtype: torch.dtype = None,
    limit_size=500 * 1024 * 1024,
):
    """
    Converts all the txt files into a pytorch file containing the ids of the
    tensors (in format [1, t] where t is the number of tokesn)

    Args:
        txt_file_path: the location of the source text file
        pt_file_path: the location of the destination pytorch token file
        tokenizer_name (optional): if there is a tokenizer with that name in
            tokenizers_by_name
        m_name (optional): a HuggingFace tokenizer name e.g. 'gpt2',
            'gpt2-large', 'facebook/opt-125m', or 'bert-base-uncased'
        m_path (optional): the path to the (HuggingFace) tokenizer (saved
            offline)
        dtype (optional): by default, the token tensors are int64, but another
            type (such as int16) can be specified
    """
    tokenizer = get_tokenizer(
        tokenizer_name=tokenizer_name, m_name=m_name, m_path=m_path
    )
    tokenize_dir_txt_files_from_tok(
        dir_path, pt_file_path, tokenizer=tokenizer, dtype=dtype
    )

    # if pt_file_path.endswith('.pt'): pt_file_path = pt_file_path[:-len('.pt')]
    # if pt_file_path.endswith('.txt'): pt_file_path = pt_file_path[:-len('.txt')]

    # txt_file_paths = []
    # for (subdir, _, files) in os.walk(dir_path):
    #     txt_file_paths.extend([os.path.join(subdir, file) for file in files if file.endswith('.txt')])

    # part_index = 0
    # big_txt = '' # A concatenation of the files

    # def save_and_clear_big_txt(big_txt, part_index):
    #     part_txt_file_path = pt_file_path + f'_{part_index:04}_bigtext.txt'
    #     save_string(big_txt, part_txt_file_path)
    #     part_pt_file_path = pt_file_path + f'_{part_index:04}.pt'
    #     tokenizer.tokenize_txt_file_to_pt_file(part_txt_file_path, part_pt_file_path, dtype=dtype)
    #     os.remove(part_txt_file_path)
    #     log(f"Saved and cleared part {part_index:04}")

    # print("Starting tokenization of files")
    # for txt_file_path in tqdm(txt_file_paths):
    #     print(f"Adding {txt_file_path} to big_txt")
    #     txt = load_string(txt_file_path) + ('\n' * 5)

    #     big_txt += txt
    #     while len(big_txt) >= limit_size:
    #         text_chunk = big_txt[:limit_size]
    #         print(f"Sample : {text_chunk[:100]}")
    #         save_and_clear_big_txt(text_chunk, part_index)
    #         big_txt = big_txt[limit_size:] # removes the first limit_size chars from big_txt
    #         print(f"Split big_txt into a chunk of {limit_size/1e6:.2f}Mchar")
    #         part_index += 1
    #     print(f"Full big_txt length : {len(big_txt)}")
    # save_and_clear_big_txt(big_txt, part_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Tokenize .txt files in a given directory into PyTorch tensors of the
        given type, using a specific tokenizer.

        Use:
            python tokenizer.py <txt_files_dir> <tokenizer_name> <dtype:uint8,visuint16,int32,int64>
        """
    )

    parser.add_argument(
        "input_directory",
        type=str,
        help="""
        The directory path containing the txt files.
        """,
    )

    # TODO: merge both into one arg as AutoTokenizer works for either

    toks = parser.add_mutually_exclusive_group()

    toks.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="""
        The name of the Huggingface-downloadable tokenizer to
        use. Default: 'gpt2'.
        """,
    )

    toks.add_argument(
        "--tokenizer_local",
        type=str,
        default=None,
        help="""
        The name of the locally saved tokenizer to use.
        """,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["uint8", "visuint16", "int32", "int64"],
        default=None,
        help="""
        The dtype to cast the PyTorch tensors into. Choices:
        'uint8,visuint16,int32,int64'. Default: None.
        """,
    )

    args = parser.parse_args()

    print(
        f"Received: dir: {args.input_directory}, model:{args.tokenizer}, dtype :{args.dtype}"
    )

    tokenize_dir_txt_files(
        dir_path=args.input_directory,
        pt_file_path=f"{args.input_directory}.pt",
        m_name=args.tokenizer,
        m_path=args.tokenizer_local,
        dtype=args.dtype,
    )
