"""
Defines the methods needed for tokenizing of .txt files into .pt, given
a pre-existing tokenizer.

Can be used as a script.

Usage:
    python tokenize_to_pt.py <folder_path> -t <tokenizer_name> -p

Arguments:
    folder_path (str): Mandatory. Path to the folder containing txt files to be
        tokenized.
    --tokenizer_name, -t (str): Optional. Specifies the tokenizer by name.
        Defaults to 'gpt2' if not provided.
    --no_preprocess, -p: Optional flag. If set, skips splitting and sanitizing
        the text files.

The script processes each text file in the directory with the specified or
default tokenizer, and can optionally skip preprocessing steps based on the
command line arguments.
"""

import os
import shutil
import argparse
from tqdm import tqdm

import torch

from .. import tokenizer

MAX_SIZE = 2 * 1024 * 1024 * 1024


def replace_unusual_terminators(filename):
    """Replace unusual line terminators with standard newline."""
    LS = "\u2028"
    PS = "\u2029"

    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

        data = data.replace(LS, "\n").replace(PS, "\n")

        with open(filename, "w", encoding="utf-8") as f:
            f.write(data)


def main_replace_unusual(folder_path):
    print("Replacing terminators ...")
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            replace_unusual_terminators(filepath)

            # print(f"Replaced terminators in {filename}")


def split_file(filename, split_dir):
    """Split a file into multiple 500MB parts while ensuring split occurs at a newline."""
    print(f"Splitting {filename}")
    part_size = MAX_SIZE  # 500MB
    part_num = 1

    with open(filename, "rb") as source:
        while True:
            # Read up to the desired part size
            data = source.read(part_size)
            if not data:
                break

            # Check if there's more data after this to determine if we should look for a newline
            while True:
                buffer = source.peek(1024 * 4)
                if not buffer:
                    break
                newline_pos = buffer.find(b"\n")
                if newline_pos != -1:
                    data += source.read(newline_pos + 1)  # Only read up to the newline
                    break
                # Read the entire peeked buffer since no newline was found
                data += source.read(1024 * 4)

            # part_file = f"{filename[:-4]}_{part_num:04}.txt"
            part_file = os.path.join(
                split_dir,
                os.path.splitext(os.path.basename(filename))[0],
                f"_{part_num:04}.txt",
            )
            with open(part_file, "wb") as target:
                target.write(data)
            print(f"Split {part_num*MAX_SIZE/1e9}GB so far")
            part_num += 1


def main_split_large(folder_path, target_path):
    """
    Use split_file on a folder containing big .txt files.
    Backs up the un-split text.

    Args:
        folder_path: Path to the folder containing the .txt files to split.
        target_path: Path to the folder that will contain the splits (created
            if needed).
    """
    os.makedirs(target_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        print(f"Scanning and splitting {os.path.join(folder_path,filename)}")

        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            if os.path.getsize(filepath) > MAX_SIZE + 64 * 1024:
                # Split file to target folder
                split_file(filepath, target_path)
            else:
                # Copy original file to split folder
                shutil.copy(filepath, target_path)


def tokenize_folder(folder_path, tokenizer_path=None, no_preprocess=False):
    """
    Pipeline for tokenizing text in a folder. Is NOT recursive, will
    act only on .txt files contained in folder_path.

    Save the tensors containing the tokens as .pt files in a folder named
    `folder_path`_pt (appending "_pt"). Each .pt file contains a tensor of
    shape [1, n_tokens], which can be loaded with torch.load if needed.

    Args:
        folder_path: Path to the folder containing the .txt files to
            tokenize.
        tokenizer_path: Path to the tokenizer to use. If None, will use the
            default GPT2 tokenizer.
        no_preprocess: If True, will not do the splitting and sanitization
            of the files. Default is False. (use True if tokenization
            crashed after sanitization, to not repeat preprocessing)
    """
    if tokenizer_path is None:
        raise ValueError(
            "Tokenizer path required, please specify with -t <tokenizer_path>"
        )
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at path {tokenizer_path}. \nTokenizer path should be relative current folder."
        )
    else:
        tokenizer_name = os.path.basename(tokenizer_path)

    toki = tokenizer.get_tokenizer(m_path=tokenizer_path, m_name=tokenizer_name)

    target_path = f"{folder_path}_pt"

    if not no_preprocess:
        # Only do if no_preprocess is false
        main_split_large(folder_path, target_path)  # First split into MAX-SIZE files
        # Then remove strange terminators
        main_replace_unusual(target_path)

    # Then run the tokenizer on the MAX_SIZE txt files :
    for txtfile in os.listdir(target_path):
        if txtfile.endswith(".txt"):
            toki.tokenize_txt_file_to_pt_file(
                os.path.join(target_path, txtfile),
                f"{os.path.join(target_path,txtfile[:-4])}_tokenized.pt",
                dtype=torch.int32,
            )
            # Delete the txt files once done, they are backed-up anyway
            os.remove(os.path.join(target_path, txtfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split large txt files in a directory."
    )

    parser.add_argument("folder_path", help="Path to the folder containing txt files.")

    # TODO: Add argument for tokenizer location
    parser.add_argument(
        "--tokenizer_path", "-t",
        help="""
        Relative path containing saved tokenizer to use. Mandatory parameter.
        """,
    )

    parser.add_argument(
        "--no_preprocess", "-p",
        help="""
        If specified, does not do the splitting and sanitization of the files
        """,
        action="store_true",
    )

    args = parser.parse_args()

    print(
        f"Tokenizing {args.folder_path} with tokenizer in path : {args.tokenizer_path}"
    )

    tokenize_folder(
        args.folder_path,
        tokenizer_name=args.tokenizer_path,
        no_preprocess=args.no_preprocess,
    )
