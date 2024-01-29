import h5py , os, torch
from pathlib import Path
from tqdm import tqdm
from modules import tokenizer
from torch.utils.data import Subset
import numpy as np

toki = tokenizer.get_tokenizer(m_path='modules/tokenizers/fr_tokenizer',m_name='fr')
def make_h5(pt_data_folder, destination_folder = None):
    if(destination_folder is None):
        destination_folder= Path(__file__).parent.as_posix()
    
    tarname = os.path.join(destination_folder,f'{os.path.basename(pt_data_folder)}.h5')
    os.makedirs(os.path.dirname(tarname),exist_ok=True)


    if(os.path.isdir(pt_data_folder)):
        with h5py.File(tarname, 'w') as f:
            dset = f.create_dataset("tokens", (0,), maxshape=(None,), dtype='int32')  # note the maxshape parameter
            
            current_index = 0
            for file in tqdm(os.listdir(pt_data_folder)):
                if os.path.splitext(file)[1]=='.pt':
                    pt_file = os.path.join(pt_data_folder,file)
                    tensor = torch.load(pt_file,map_location=torch.device('cpu'))
                    length = tensor.shape[1]
                    print('snippet', toki.detokenize(tensor[:,:40]))
                    # Resize the dataset to accommodate the new data
                    dset.resize((current_index + length,))
                    
                    # Add the new data to the dataset
                    dset[current_index:current_index+length] = tensor.numpy().squeeze()
                    
                    # Update the current ind
                    current_index += length
    else :
        raise ValueError(f'{pt_data_folder} not found')



def make_h5_from_dataset(dataset,name, toki_test, destination_folder= None, concat_num_before_write=100000):
    """
        Saves the TOKENTEXTBOS dataset as a h5 file of consecutive tokens. WARNING ! Can be reused ONLY with the same
        attention length, otherwise it will be garbage text. Indeed, it will concatenate all of the sentences, so it must
        be re-cut with the exact same way to not obtain garbage.
    """
    if(destination_folder is None):
        destination_folder= os.path.join(Path(__file__).parent.as_posix(),'datavol','scrambled')
    
    if(isinstance(dataset,Subset)):
        attn_length = dataset.dataset.attn_length+1
    else :
        attn_length = dataset.attn_length+1
    
    tarname = os.path.join(destination_folder,f'{name}_attn{attn_length}.h5')
    os.makedirs(destination_folder,exist_ok=True)



    with h5py.File(tarname, 'w') as f:
        dset = f.create_dataset("tokens", (0,), maxshape=(None,), dtype='int32')  # note the maxshape parameter
        
        current_index = 0
        tensor = torch.zeros((0,),dtype=torch.int32)
        for _,answer in tqdm(dataset):
            # Answer has shape (attn_length), containing the full phrase since its tokentextbos
            tensor = [tensor,answer]
            if(len(tensor)>=concat_num_before_write):
                length=len(tensor)
                
                # Resize the dataset to accommodate the new data
                dset.resize((current_index + length,))
                tensor = torch.cat(tensor,dim=0)
                # Add the new data to the dataset
                dset[current_index:current_index+length] = tensor.numpy()
                print('snippet', toki_test.detokenize(tensor[:60]))
                # Update the current ind
                current_index += length
                tensor = []        
        # Last one
        length =tensor.shape[0]
        if(length>0):
            print('snippet', toki.detokenize(tensor[:,:40]))
            dset.resize((current_index + length,))
            dset[current_index:current_index+length] = tensor.numpy()
    
    print(f'SAVED scrambled {tarname} !')

def split_h5(h5_path, destination_folder, dest_name=None, split_parts=3, num_tokens_together = 100000):
    """
        Splits the h5 file in split_parts parts, and saves them in destination_folder.
    """
    os.makedirs(destination_folder,exist_ok=True)
    h5_file = h5py.File(h5_path, 'r')
    text_tensor = h5_file['tokens']

    num_splits = split_parts
    split_size = int(text_tensor.shape[0]/num_splits) # we miss a few tokens, but its ok

    if(dest_name is None):
        dest_name = os.path.basename(h5_path).split('.')[0]
    for k in tqdm(range(num_splits)):
        split_path = os.path.join(destination_folder,f'{dest_name}_{k}.h5')
        with h5py.File(split_path, 'w') as f2:
            dset2 = f2.create_dataset("tokens", (0,), maxshape=(None,), dtype='int32')  # note the maxshape parameter
            start = k*int(split_size)
            end = min((k+1)*int(split_size),text_tensor.shape[0])
            length = end-start
            dset2.resize((length,))

            location = start
            while location<=end-num_tokens_together:
                dset2[location-start:location-start+num_tokens_together] = text_tensor[location:location+num_tokens_together]
                location+=num_tokens_together
        
            dset2[location-start:end-start] = text_tensor[location:end]

            print(f'Saved {split_path} !')

if __name__=='__main__':
    make_h5('testdata', destination_folder='test_h5')
