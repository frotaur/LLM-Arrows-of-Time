"""
    All the pipeline to tokenize big files
"""
from .. import tokenizer
import  os, torch, argparse, shutil,pathlib
from tqdm import tqdm

MAX_SIZE = 2*1024*1024*1024 

def replace_unusual_terminators(filename):
    """Replace unusual line terminators with standard newline."""
    LS = '\u2028'
    PS = '\u2029'

    with open(filename, 'r', encoding='utf-8',errors='ignore') as f:
        data = f.read()
    
        data = data.replace(LS, '\n').replace(PS, '\n')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(data)

def main_replace_unusual(folder_path):
    print('Replacing terminators ...')
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            replace_unusual_terminators(filepath)


def split_file(filename):
    """Split a file into multiple 500MB parts while ensuring split occurs at a newline."""
    print(f'Splitting {filename}')
    part_size = MAX_SIZE  # 500MB
    part_num = 1

    counter = 0
    with open(filename, 'rb') as source:
        while True:
            # Read up to the desired part size
            data = source.read(part_size)
            if not data:
                break

            # Check if there's more data after this to determine if we should look for a newline
            while True:
                buffer = source.peek(1024*4)
                if not buffer:
                    break
                newline_pos = buffer.find(b'\n')
                if newline_pos != -1:
                    data += source.read(newline_pos + 1)  # Only read up to the newline
                    break
                data += source.read(1024*4)  # Read the entire peeked buffer since no newline was found


            part_file = f"{filename[:-4]}_{part_num:04}.txt"
            with open(part_file, 'wb') as target:
                target.write(data)
            print(f'Splitted {part_num*MAX_SIZE/1e9}GB so far')
            part_num += 1
            

def main_split_large(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    backup_dir = os.path.join(folder_path,'..',f'{folder_name}_backup')
    os.makedirs(backup_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        print(f'Scanning and splitting {os.path.join(folder_path,filename)}')

        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            if os.path.getsize(filepath) > MAX_SIZE+64*1024:
                split_file(filepath)
                # Move original file to backup folder
                shutil.move(filepath, backup_dir)
            else:
                # Copy original file to backup folder
                shutil.copy(filepath, backup_dir)
            
            

def separate_dataset(folder_path):
    """
    Separate a folder of .txt files into sub-datasets of each max 20GB.
    Not necessary anymore now that we can tokenize file by file.
    """
    sep_num = 0
    folder_name = os.path.basename(os.path.normpath(folder_path))
    cur_path = os.path.join(folder_path, f'{folder_name}_{sep_num:03d}')
    print(f'Separate with {folder_path=}')
    to_move = []
    print('Separating into smaller datasets...')
    for filename in os.listdir(folder_path):
        if(filename.endswith('.txt')):
            to_move.append(os.path.join(folder_path, filename))
        if(len(to_move)>=10):
            print(f'Make a dir at {cur_path=}')
            os.makedirs(cur_path, exist_ok=True)
            for file in to_move:
                shutil.move(file, cur_path)
            sep_num += 1
            cur_path = os.path.join(folder_path, f'{folder_name}_{sep_num:03d}')
            to_move = []
    
    if(len(to_move)>0):
        os.makedirs(cur_path, exist_ok=True)
        for file in to_move:
            shutil.move(file, cur_path)

"""
    All the pipeline to tokenize big files
"""
from modules import tokenizer
import  os, torch, argparse, shutil,pathlib
from tqdm import tqdm

MAX_SIZE = 2*1024*1024*1024 

def replace_unusual_terminators(filename):
    """Replace unusual line terminators with standard newline."""
    LS = '\u2028'
    PS = '\u2029'

    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    
        data = data.replace(LS, '\n').replace(PS, '\n')

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(data)

def main_replace_unusual(folder_path):
    print('Replacing terminators ...')
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            replace_unusual_terminators(filepath)

                # print(f"Replaced terminators in {filename}")

def split_file(filename):
    """Split a file into multiple 500MB parts while ensuring split occurs at a newline."""
    print(f'Splitting {filename}')
    part_size = MAX_SIZE  # 500MB
    part_num = 1

    counter = 0
    with open(filename, 'rb') as source:
        while True:
            # Read up to the desired part size
            data = source.read(part_size)
            if not data:
                break

            # Check if there's more data after this to determine if we should look for a newline
            while True:
                buffer = source.peek(1024*4)
                if not buffer:
                    break
                newline_pos = buffer.find(b'\n')
                if newline_pos != -1:
                    data += source.read(newline_pos + 1)  # Only read up to the newline
                    break
                data += source.read(1024*4)  # Read the entire peeked buffer since no newline was found


            part_file = f"{filename[:-4]}_{part_num:04}.txt"
            with open(part_file, 'wb') as target:
                target.write(data)
            print(f'Splitted {part_num*MAX_SIZE/1e9}GB so far')
            part_num += 1
            

def main_split_large(folder_path):
    folder_name = os.path.basename(os.path.normpath(folder_path))
    backup_dir = os.path.join(folder_path,'..',f'{folder_name}_backup')
    os.makedirs(backup_dir, exist_ok=True)

    for filename in os.listdir(folder_path):
        print(f'Scanning and splitting {os.path.join(folder_path,filename)}')

        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            if os.path.getsize(filepath) > MAX_SIZE+64*1024:
                split_file(filepath)
                # Move original file to backup folder
                shutil.move(filepath, backup_dir)
            else:
                # Copy original file to backup folder
                shutil.copy(filepath, backup_dir)
            
            

def separate_dataset(folder_path):
    """
    Separate a folder of .txt files into sub-datasets of each max 20GB.
    Not necessary anymore now that we can tokenize file by file.
    """
    sep_num = 0
    folder_name = os.path.basename(os.path.normpath(folder_path))
    cur_path = os.path.join(folder_path, f'{folder_name}_{sep_num:03d}')
    print(f'Separate with {folder_path=}')
    to_move = []
    print('Separating into smaller datasets...')
    for filename in os.listdir(folder_path):
        if(filename.endswith('.txt')):
            to_move.append(os.path.join(folder_path, filename))
        if(len(to_move)>=10):
            print(f'Make a dir at {cur_path=}')
            os.makedirs(cur_path, exist_ok=True)
            for file in to_move:
                shutil.move(file, cur_path)
            sep_num += 1
            cur_path = os.path.join(folder_path, f'{folder_name}_{sep_num:03d}')
            to_move = []
    
    if(len(to_move)>0):
        os.makedirs(cur_path, exist_ok=True)
        for file in to_move:
            shutil.move(file, cur_path)

def tokenize_folder(folder_path, tokenizer_path=None, no_preprocess=False):
    """
        Pipeline for tokenizing text in a folder. Is NOT recursive, will
        act only on .txt files contained in folder_path.

        Args:
        folder_path : Path to the folder containing the .txt files to tokenize.
        tokenizer_path : Path to the tokenizer to use. If None, will use the default GPT2 tokenizer.
        no_preprocess : If True, will not do the splitting and sanitization of the files. Default is False. (use True if tokenization crashed after sanitization, to not repeat it)
    """
    if(tokenizer_path is None):
        print("CAUTION ! USING DEFAULT GPT2")
        tokenizer_name = 'gpt2'
    else :
        tokenizer_name = os.path.basename(tokenizer_path)

    toki = tokenizer.get_tokenizer(m_path=tokenizer_path,m_name=tokenizer_name)
    if(not no_preprocess):
        # Only do if no_preprocess is false
        main_split_large(folder_path)# First split into MAX-SIZE files
        # Then remove strange terminators
        main_replace_unusual(folder_path)

    # # Then separate into 20GB datasets
    # separate_dataset(folder_path)

    # Then run the tokenizer on the MAX_SIZE txt files : 
    for txtfile in os.listdir(folder_path):
        if(txtfile.endswith('.txt')):
            toki.tokenize_txt_file_to_pt_file(os.path.join(folder_path,txtfile), f'{os.path.join(folder_path,txtfile[:-4])}_tokenized.pt', dtype=torch.int32)
            # Delete the txt files once done, they are backed-up anyway
            os.remove(os.path.join(folder_path,txtfile))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split large txt files in a directory.")
    parser.add_argument("folder_path", help="Path to the folder containing txt files.")
    #Add argument for tokenizer location
    parser.add_argument("--tokenizer_name","-t", help="Named tokenizer to use (name of folder in modules/tokenizers). If not specified, will use gpt2 tokenizer.")
    parser.add_argument("--no_preprocess","-p",help="If specified, does not do the splitting and sanitization of the files", action="store_true")
    args = parser.parse_args()

    print(f'Tokenizing {args.folder_path} with {args.tokenizer_name} tokenizer')
    tokenize_folder(args.folder_path,tokenizer_name=args.tokenizer_name,no_preprocess=args.no_preprocess)