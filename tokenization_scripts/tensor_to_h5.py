import argparse, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modules.tok_utils import make_h5
from modules.tokenizer import get_tokenizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Converts saved PyTorch tensors (.pt) into an .h5 dataset.
        """
    )

    parser.add_argument(
        "input_directory",
        type=str,
        help="""
        Path to the folder containing the .pt files. Can be
        relative or absolute.
        """,
    )

    parser.add_argument(
        "--output_directory",
        "-d",
        type=str,
        help="""
        Path to the folder where the .h5 file will be saved.
        Can be relative or absolute.
        """,
    )

    parser.add_argument(
        "--dataset_name",
        "-n",
        type=str,
        help="""
        Name of the dataset file to be produced. Defaults to
        `dataset.h5` ('.h5' will automatically be added at the
        end if missing).
        """,
    )

    # TODO: implement the choice of tokenizer (HF or local)
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        help="""
        Tokenizer folder to use for viewing dataset snippets during
        conversion. Optional
        """
    )

    args = parser.parse_args()

    tokenizer = get_tokenizer(m_path=args.tokenizer) if args.tokenizer else None
    make_h5(
        pt_data_folder=args.input_directory,
        dataset_fname=args.dataset_name,
        destination_folder=args.output_directory,
        view_tokenizer=tokenizer
    )
