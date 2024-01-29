from modules import *
import torch, torch.optim,os, argparse,json, pathlib,random, shutil
from torch.utils.data import Subset
from torchenhanced import CosineWarmup
import numpy as np, pickle
from tqdm import tqdm


rng = np.random.default_rng(42) # For deterministic shuffling of dataset
# First copy the dataset in the current folder. This is useful in the case of a network drive, where the dataset is slow to access.
# Can be removed if not used on Runai.
dataset_path = 'datavol/vassilis/french/french.h5'

destination_path = os.path.join('.','local_dataset.h5')
dataname = os.path.splitext(os.path.basename(dataset_path))[0]
if(not os.path.exists(destination_path)):
    print('Copying dataset to current folder...')
    # Use shutil.copy() to copy the file
    shutil.copy(dataset_path, destination_path)
else :
    print('Dataset already copied to current folder, using this one.')

motherDataset = TokenTextBOS(h5_file=destination_path, attn_length=16, backwards=False) 
indices_path = os.path.join(os.path.dirname(dataset_path),f'{dataname}_indices_{motherDataset.attn_length}.pkl')

if(not os.path.exists(indices_path)):
    # If the indices have not already been generated
    print('Generating indices for dataset...')
    indices = np.arange(len(motherDataset))
    rng.shuffle(indices)
    with open(indices_path,'wb') as f:
        pickle.dump(indices,f)
else:
    ## TO CORRECT WHAT TO DO WHEN WE NEED TO LOAD MORE TAHN ONE INDEX
    with open(indices_path,'rb') as f:
        print('Loading indices from',indices_path)
        indices = pickle.load(f)