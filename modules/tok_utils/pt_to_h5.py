import h5py , os, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def make_h5(pt_data_folder, destination_folder = None, view_tokenizer : AutoTokenizer = None):
    """
        Takes a folder of .pt files, and converts them to a single .h5 file.

        Args:
            pt_data_folder : Path to the folder containing the .pt files. Can be relative or absolute.
            destination_folder : Path to the folder where the .h5 file will be saved. Can be relative or absolute.
            view_tokenizer : A tokenizer to use for viewing dataset snippets during conversion. If None, will use the GPT2 tokenizer.
    """

    if(view_tokenizer is None):
        view_tokenizer = AutoTokenizer.from_pretrained('gpt2',use_fast=True)

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

                    print('snippet : \n', view_tokenizer.detokenize(tensor[:,:40]))
                    # Resize the dataset to accommodate the new data
                    dset.resize((current_index + length,))
                    
                    # Add the new data to the dataset
                    dset[current_index:current_index+length] = tensor.numpy().squeeze()
                    
                    # Update the current ind
                    current_index += length
    else :
        raise ValueError(f'{pt_data_folder} not found')


if __name__=='__main__':
    make_h5('testdata', destination_folder='test_h5')
