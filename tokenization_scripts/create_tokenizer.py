import argparse, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modules.tok_utils import create_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Trains a Huggingface Tokenizer with BPE.
        """
    )

    parser.add_argument(
        "input_file_or_folder",
        type=str,
        help="""
        Path to the .txt file to use for training the tokenizer.
        Can also be a folder containing multiple .txt files.
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
        txt_path=args.input_file_or_folder,
        save_directory=args.output_directory,
        tokenizer_name=args.tokenizer_name,
        vocab_size=args.vocab_size,
    )
