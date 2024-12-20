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
from pathlib import Path

def txt_to_h5(txt_path, out_h5_folder, tokenizer_folder, tokenizer_name):
    """
    Given a folder containing .txt files, trains a BPE tokenizer on
    it. Then, tokenizes the .txt files, and save the result as a an .h5 file,
    which can be used to make a TokenTextBOS dataset.

    NOTE: There are NO checkpoints, if it crashes at any point, you have to
    start over. To avoid this, use instead the scripts 'create_tokenizer',
    'tok_txt_to_tensor' and 'tensor_to_h5' separately.

    Args:
        txt_path (str or Path): Folder containing .txt files
        out_h5_folder (str or Path): Folder to the output .h5 file
        tokenizer_folder (str or Path): Folder where the tokenizer will be saved
        tokenizer_name (str or Path): Name of the tokenizer that will be saved
    """
    txt_path = Path(txt_path)
    out_h5_folder = Path(out_h5_folder)
    tokenizer_folder = Path(tokenizer_folder)

    create_tokenizer(txt_path, save_directory=tokenizer_folder, tokenizer_name=tokenizer_name)
    
    tokenize_folder(
        txt_path, tokenizer_path= tokenizer_folder / tokenizer_name
    )

    toki = get_tokenizer(m_path=(tokenizer_folder / tokenizer_name).as_posix())

    
    pt_data_folder = txt_path.parent / f"{txt_path.name}_pt"

    make_h5(
        pt_data_folder=pt_data_folder,
        dataset_fname=txt_path.name,
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
        The input folder to be tokenized. This script will save the following items:
        1) A folder named '<txt_folder>_pt', containing the tokenized data as pytorch tensors. A folder named '<txt_folder>_h5' containing the tokenized h5py dataset. Example:
            my_dataset/input.txt -> my_dataset_h5/input.h5
                                    my_dataset_pt/input_tokenized.pt
        2) a tokenizer in modules/tokenizers named '<txt_folder>_tokenizer'. Example:  
        code_dataset/input.txt -> modules/tokenizers/code_dataset_tokenizer/
        """,
    )

    args = parser.parse_args()

    txt_folder_full = Path(args.txt_folder)
    txt_folder_name = Path(txt_folder_full).name

    txt_folder_parent = Path(txt_folder_full).parent

    out_h5_folder = txt_folder_parent / f"{txt_folder_name}_h5"
    tokenizer_folder = Path("modules/tokenizers")  # Folder where the tokenizer will be saved
    # Name of the tokenizer that will be saved
    tokenizer_name = f"{txt_folder_name}_tokenizer"

    ################## DO NOT MODIFY BELOW ##################
    txt_to_h5(txt_folder_full, out_h5_folder, tokenizer_folder, tokenizer_name)
