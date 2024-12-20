import argparse, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modules.tok_utils import tokenize_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenizes all .txt files in a folder with a given tokenizer."
    )

    parser.add_argument("folder_path", help="Path to the folder containing txt files.")
    parser.add_argument(
        "--tokenizer_path", "-t",
        help="""
        Relative path containing saved tokenizer to use. Mandatory parameter.
        """,
        required=True,
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
        tokenizer_path=args.tokenizer_path,
        no_preprocess=args.no_preprocess,
    )
