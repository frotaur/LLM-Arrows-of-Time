"""
Contains the method necessary to train a Tokenizer with BPE. Only work on a
single .txt file. Should be no bigger than ~50 GB, to avoid memory issues.

File should contain regular spaces, otherwise it might crash with OOM error.
"""

import os
import argparse

from transformers import AutoTokenizer


# Default chunk size is 10MB
def read_in_chunks(
    file_path, chunk_size=10 * 1024 * 1024
):  # Default chunk size is 10MB
    """
    Generator to yield chunks of text from a large file.
    TODO : Add support for .txt files which are not unified
    """
    counter = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:  # End of file
                break
            counter += 1
            if counter % 50 == 0:
                print(f"Processed {counter*chunk_size/(1024*1024*1024):.1f}GB")
            yield chunk


def read_in_lines(
    file_path, batch_size=512, phrase_size=2048
):  # phrase_size is approx number of characters in a element
    """
    Generator to yield chunks of text from a large file.
    TODO : Add support for .txt files which are not unified
    """
    counter = 0
    total_line_weight = 0

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        done = False
        while not done:
            lines = []
            newline = ""

            while len(lines) < batch_size and not done:
                while len(newline) <= phrase_size:
                    # Read lines until we get something long enough
                    extra = file.readline()
                    if not extra:  # End of file
                        done = True
                        print("YAY DONE")
                        break
                    newline = newline + extra

                if len(newline) > 1.5 * phrase_size:
                    # In case of big overshoot, split the line
                    while len(newline) > phrase_size:
                        lines.append(newline[:phrase_size])
                        newline = newline[phrase_size:]

                # Append the newline, or what's left of it
                lines.append(newline)
                newline = ""

            total_line_weight += len("\n".join(lines).encode("utf-8"))  # in bytes
            counter += 1
            if counter % 300 == 0:
                print(f"Processed ~{total_line_weight/(1024*1024*1024):.1f}GB\n")

            yield lines


def create_tokenizer(
    txt_path, save_directory=None, tokenizer_name=None, vocab_size=50257
):
    """
    Creates a custom BPE huggingface tokenizer from a .txt file. The tokenizer
    is saved as a folder, and can be loaded with the helper function
    `get_tokenizer(m_path=<tokenizer folder>)` from modules/tokenizer.py. Needs
    quite a bit of memory, depending on dataset size. If the data has no
    spaces, it will probably crash with out of memory error.

    Args:
        txt_path: Path to the .txt file to use for training the tokenizer.
        save_directory: Directory where the tokenizer will be saved. If None,
            will be saved in the same directory as the .txt file.
        tokenizer_name: Name of the tokenizer. If None, will be the name of the
            .txt file, followed by `_tokenizer`.
        vocab_size: Size of the vocabulary to use for the tokenizer. Default is
            50257, which is the GPT2 vocabulary size.
    """
    if save_directory is None:
        save_directory = os.path.dirname(txt_path)

    if tokenizer_name is None:
        tokenizer_name = f'{os.path.basename(txt_path).split(".")[0]}_tokenizer'
        print(f"No tokenizer name provided, defaulting to: {tokenizer_name}")

    # Load gpt2 tokenizer, to get same vocab_size and special tokens
    toke_base = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    # .txt dataset (full dataset in the txt file, for now.)
    toke_mine = toke_base.train_new_from_iterator(
        read_in_lines(txt_path), vocab_size=vocab_size
    )  # Train new tokenizer using BPE
    toke_mine.save_pretrained(os.path.join(save_directory, tokenizer_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Trains a Huggingface Tokenizer with BPE. Only work on a single .txt
        file. Should be no bigger than ~50 GB, to avoid memory issues.
        """
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="""
        Path to the .txt file to use for training the tokenizer.
        """,
    )

    parser.add_argument(
        "--output_directory", "-d",
        type=str,
        help="""
        Directory where the tokenizer will be saved. If None,
        will be saved in the same directory as the .txt file.
        """,
    )

    parser.add_argument(
        "--tokenizer_name", "-t",
        type=str,
        default=None,
        help="""
        Name of the tokenizer. If None, will be the name of the
        .txt file, followed by `_tokenizer`. Default is None.
        """,
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=50257,
        help="""
        Size of the vocabulary to use for the tokenizer.
        Default is 50257, which is the GPT2 vocabulary size.
        """,
    )

    args = parser.parse_args()

    create_tokenizer(
        txt_path=args.input_file,
        save_directory=args.output_directory,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
    )
