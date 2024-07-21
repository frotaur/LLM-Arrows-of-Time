"""
Script for preparing a .txt cc-100 dataset for training.

Creates the custom tokenizer, and tokenizes the text with it to generate the
.h5 file for training.

To make one of those things independently (e.g., only make the custom
tokenizer), see modules/tok_utils.
"""
import os
import argparse

from modules.tok_utils import make_h5
from modules.tok_utils import tokenize_folder
from modules.tok_utils import create_tokenizer

from modules import get_tokenizer




def txt_to_h5(txt_path, out_h5_folder, tokenizer_folder, tokenizer_name):
    """
    Given a .txt file located ALONE inside a folder, trains a BPE tokenizer on
    it. Then, tokenizes the .txt file, and save the result as a an .h5 file,
    which can be used to make a TokenTextBOS dataset.

    NOTE: There are NO checkpoints, if it crashes at any point, you have to
    start over. To avoid this, use instead the functions 'create_tokenizer',
    'tokenize_folder' and 'make_h5' separately.

    Args:
        txt_path (str): Path to the (single) .txt file to be tokenized
        out_h5_folder (str): Folder to the output .h5 file
        tokenizer_folder (str): Folder where the tokenizer will be saved
        tokenizer_name (str): Name of the tokenizer that will be saved
    """
    create_tokenizer(txt_path, tokenizer_folder,tokenizer_name=tokenizer_name)
    tokenize_folder(
        os.path.dirname(txt_path), os.path.join(tokenizer_folder, tokenizer_name)
    )
    toki = get_tokenizer(m_path=os.path.join(tokenizer_folder, tokenizer_name))
    make_h5(
        os.path.dirname(txt_path),
        os.path.splitext(os.path.basename(txt_path))[0],
        out_h5_folder,
        toki,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script for preparing a .txt cc-100 dataset for training. Creates the
        custom tokenizer, and tokenizes the text with it to generate the .h5
        file for training.

        To make one of those things independently (e.g., only make the custom
        tokenizer), see modules/tok_utils.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--txt_path", "-t",
        type=str,
        required=True,
        help="""
        The input file to be tokenized. This script will save the following
        items:
        1) given the path of a source plain text file, a folder of the same
        name as the containing folder of txt_path, with '_h5' appended at the
        end, as well as raw Pytorch tensors. Example:
            -t my_dataset/input.txt -> my_dataset_h5/input.h5
                                       my_dataset_pt/input_tokenized.pt
        2) a tokenizer in modules/tokenizers called after the folder containing
        the txt dataset. Example:
            -t code_dataset/input.txt -> modules/tokenizers/code_dataset_tokenizer/
        """,
    )

    args = parser.parse_args()

    txt_folder = os.path.split(args.txt_path)[0]

    out_h5_folder = f"{txt_folder}_h5"  #  Folder that will contain the output .h5 file
    tokenizer_folder = "modules/tokenizers"  # Folder where the tokenizer will be saved
    # Name of the tokenizer that will be saved
    tokenizer_name = f"{txt_folder}_tokenizer"

    ################## DO NOT MODIFY BELOW ##################
    txt_to_h5(args.txt_path, out_h5_folder, tokenizer_folder, tokenizer_name)
