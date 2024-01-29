from .. import tokenizer
import os
from transformers import AutoTokenizer

#==================== TEXT PATH ====================


def read_in_chunks(file_path, chunk_size=10*1024*1024):  # Default chunk size is 10MB
    """
    Generator to yield chunks of text from a large file.
    TODO : Add support for .txt files which are not unified
    """
    counter = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:  # End of file
                break
            counter+=1
            if(counter%50==0):
                print(f'Processed {counter*chunk_size/(1024*1024*1024):.1f}GB')
            yield chunk

def read_in_lines(file_path, batch_size=512, phrase_size=2048):  # phrase_size is approx number of characters in a element
    """
    Generator to yield chunks of text from a large file.
    TODO : Add support for .txt files which are not unified
    """
    counter = 0
    total_line_weight=0

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        done=False
        while not done:
            lines=[]
            newline=''
            
            while (len(lines)<batch_size and not done):
                while(len(newline)<=phrase_size):
                    #Read lines until we get something long enough
                    extra=file.readline()
                    if not extra:  # End of file
                        done=True
                        print('YAY DONE')
                        break
                    newline = newline+extra

                if(len(newline)>1.5*phrase_size):
                    # In case of big overshoot, split the line
                    while(len(newline)>phrase_size):
                        lines.append(newline[:phrase_size])
                        newline=newline[phrase_size:]

                # Append the newline, or what's left of it
                lines.append(newline)
                newline=''

            total_line_weight+=len('\n'.join(lines).encode('utf-8')) # in bytes
            counter+=1
            if(counter%300==0):
                print(f'Processed ~{total_line_weight/(1024*1024*1024):.1f}GB\n')

            yield lines


def create_tokenizer(txt_path, save_directory = None,vocab_size=50257):
    """
        Creates a custom BPE huggingface tokenizer from a .txt file. 
        The tokenizer is saved in a folder with the same name as the .txt file, in the same directory.
    """
    if save_directory is None:
        save_directory = os.path.dirname(txt_path)

    toke_base = AutoTokenizer.from_pretrained('gpt2',use_fast=True)
    # .txt dataset (full dataset in ht txt file, for now.)
    toke_mine= toke_base.train_new_from_iterator(read_in_lines(txt_path),vocab_size=vocab_size)
    toke_mine.save_pretrained(os.path.join(save_directory, f'{os.path.basename(txt_path).split(".")[0]}_tokenizer'))

if __name__=="__main__":
    txt_path = 'datavol/vi.txt'

    create_tokenizer(txt_path, 'vi',vocab_size=50257)