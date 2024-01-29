from modules import *
import torch, torch.optim,os, argparse,json, pathlib,random, shutil
from torch.utils.data import Subset
from torchenhanced import CosineWarmup
import numpy as np, pickle
from tqdm import tqdm


def load_indices():
    """Loads indices for given dataset and attention length"""

    rng = np.random.default_rng(42) # For deterministic shuffling of dataset
    # First copy the dataset in the current folder. This is useful in the case of a network drive, where the dataset is slow to access.
    # Can be removed if not used on Runai.
    dataset_path = 'datavol/vassilis/english/english.h5'

    attn_length=32
    indices_path = os.path.join(os.path.dirname(dataset_path),f'indices_{attn_length-1}.pkl')

    ## TO CORRECT WHAT TO DO WHEN WE NEED TO LOAD MORE TAHN ONE INDEX
    num_files= 0 
    for file in os.listdir(os.path.dirname(indices_path)):
        if file.startswith(f'indices_{attn_length-1}'):
            num_files+=1
    print(f'found {num_files} matching files')
    files = [os.path.join(os.path.dirname(indices_path),f'indices_{attn_length-1}_{k}.pkl') for k in range(num_files)]
    total_indices = []
    for file in tqdm(files):
        with open(file,'rb') as f:
            total_indices.append(pickle.load(f))
    print('loaded ! : length :  ', len(total_indices)/1e9)
    indices = np.concatenate(total_indices)
    with open(indices_path,'wb') as f:
        pickle.dump(indices,f)