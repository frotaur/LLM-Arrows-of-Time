"""
Script for preparing a .txt cc-100 dataset for training.

Creates the custom tokenizer, and tokenizes the text with it to generate the
.h5 file for training.

To make one of those things independently (e.g., only make the custom
tokenizer), see modules/tok_utils.
"""

import os, sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modules.tok_utils import make_h5
from modules.tok_utils import tokenize_folder
from modules.tok_utils import create_tokenizer

from modules import get_tokenizer


def txt_to_h5(txt_path, out_h5_folder, tokenizer_folder, tokenizer_name):
    """
    Given a folder containing .txt files, trains a BPE tokenizer on
    it. Then, tokenizes the .txt files, and save the result as a an .h5 file,
    which can be used to make a TokenTextBOS dataset.

    NOTE: There are NO checkpoints, if it crashes at any point, you have to
    start over. To avoid this, use instead the scripts 'create_tokenizer',
    'tok_txt_to_tensor' and 'tensor_to_h5' separately.

    Args:
        txt_path (str): Folder containing .txt files
        out_h5_folder (str): Folder to the output .h5 file
        tokenizer_folder (str): Folder where the tokenizer will be saved
        tokenizer_name (str): Name of the tokenizer that will be saved
    """
    create_tokenizer(txt_path, save_directory=tokenizer_folder, tokenizer_name=tokenizer_name)
    
    tokenize_folder(
        txt_path, tokenizer_path=os.path.join(tokenizer_folder, tokenizer_name)
    )

    toki = get_tokenizer(m_path=os.path.join(tokenizer_folder, tokenizer_name))
    pt_data_folder = txt_path.rstrip("/") + "_pt"
    make_h5(
        pt_data_folder=pt_data_folder,
        dataset_fname=os.path.basename(txt_path.rstrip("/")),
        destination_folder=out_h5_folder,
        view_tokenizer=toki,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Script for preparing a .txt cc-100 dataset for training. Creates the
        custom tokenizer, and tokenizes the text with it to generate the .h5
        file for training.

        To make one of those things independently (e.g., only make the custom
        tokenizer), see tokenization_scripts.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "txt_folder",
        type=str,
        help="""  
        The folder of .txt files to be tokenized. This script will save the following items:
        1) A folder named '<txt_path>_pt', containing the tokenized data as pytorch tensors. A folder named '<txt_path>_h5' containing the tokenized h5py dataset. Example:
            my_dataset/input.txt -> my_dataset_h5/input.h5
                                    my_dataset_pt/input_tokenized.pt
        2) a tokenizer in modules/tokenizers named '<txt_path>_tokenizer'. Example:  
        code_dataset/input.txt -> modules/tokenizers/code_dataset_tokenizer/
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
