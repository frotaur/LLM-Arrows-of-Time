import os
import h5py
import argparse
from tqdm import tqdm
from pathlib import Path

import torch

from .. import tokenizer

from transformers import AutoTokenizer


def make_h5(
    pt_data_folder,
    dataset_fname=None,
    destination_folder=None,
    view_tokenizer: AutoTokenizer = None,
):
    """
    Takes a folder of .pt files, and converts them to a single .h5 file.

    Args:
        pt_data_folder : Path to the folder containing the .pt files. Can be
            relative or absolute.
        dataset_fname : Name of the dataset file. Defaults to `dataset.h5`
            ('.h5' will automatically be added at the end if missing).
        destination_folder : Path to the folder where the .h5 file will be
            saved. Can be relative or absolute.
        view_tokenizer : A tokenizer to use for viewing dataset snippets
            during conversion. If None, will use the GPT2 tokenizer.
    """

    if view_tokenizer is None:
        view_tokenizer = tokenizer.get_tokenizer(m_name="gpt2")
        print("""Warning, using GPT-2 tokenizer to view tokens. 
              If the tokenizer used to tokenize the data is different,
              the tokens will not be displayed correctly. H5-ization will
              still work correctly.""")

    if destination_folder is None:
        destination_folder = '.'

    if dataset_fname is None:
        dataset_fname = "dataset.h5"
    
    if not dataset_fname.endswith(".h5"):
        dataset_fname = f"{dataset_fname}.h5"

    tarname = os.path.join(destination_folder, dataset_fname)
    os.makedirs(os.path.dirname(tarname), exist_ok=True)

    if os.path.isdir(pt_data_folder):
        with h5py.File(tarname, "w") as f:
            # note the maxshape parameter
            dset = f.create_dataset(
                "tokens", (0,), maxshape=(None,), dtype="int32"
            )  # note the maxshape parameter

            current_index = 0
            for file in tqdm(os.listdir(pt_data_folder)):
                if os.path.splitext(file)[1] == ".pt":
                    pt_file = os.path.join(pt_data_folder, file)
                    tensor = torch.load(pt_file, map_location=torch.device("cpu"))
                    length = tensor.shape[1]

                    print("snippet : \n", view_tokenizer.detokenize(tensor[:, :40]))
                    # Resize the dataset to accommodate the new data
                    dset.resize((current_index + length,))

                    # Add the new data to the dataset
                    dset[current_index : current_index + length] = (
                        tensor.numpy().squeeze()
                    )
                    print(dset[0:10])
                    # Update the current ind
                    current_index += length
    else:
        raise ValueError(f"{pt_data_folder} not found")


