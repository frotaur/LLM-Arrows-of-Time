# Training LLMs on Natural Languages
This repo contains code used for the Natural Language experiments of the paper 'Arrows of Time in Large Langugage Models'.

## Tokenization
The scrip `tokenize_to_h5.py` can be used to prepare a dataset for training. Given a .txt file, it will train a BPE tokenizer on it, then use it to tokenize the text, and save the tokenized dataset in `.h5` format.

### Usage :
To use `tokenize_to_h5.py`, first put a standalone `.txt` file inside a folder. Then, inside `tokenize_to_h5.py`, modify the following :
``` 
if __name__=='__main__':
    txt_path = '' # Path to the .txt file to be tokenized
    out_h5_folder = '' #  Folder that will contain the output .h5 file
    tokenizer_folder = '' # Folder where the tokenizer will be saved
    tokenizer_name = '' # Name of the tokenizer that will be saved
```

Then run the script. NOTE : tokenization of large .txt files (>100GB) might take a while (1,2 days). This script is NOT designed to pick up where it left off if it crashes. For bigger datasets, consider making a script (include `from modules.tok_utils import *`), and run, subsequently :
    - `create_tokenizer(txt_path, tokenizer_folder,tokenizer_name=tokenizer_name)` : Will train the BPE tokenizer on the given .txt file, and save it in <tokenizer_folder>/<tokenizer_name>
    - `tokenize_folder(os.path.dirname(txt_path), os.path.join(tokenizer_folder,tokenizer_name))` : Will tokenize the text file, splitting it into subfiles if necessary for memory reasons. Saved the tokenized tensors as `.pt`. If it crashes mid-way, can be restarted, and will pickup from last checkpoint
    - `make_h5(os.path.dirname(txt_path), out_h5_folder,toki)` : Will convert a folder containing `.pt` files into a single `.h5` dataset, ready for training.

For more informations on these functions, look at docstring comments in `modules/tok_utils`


## Training

### Scripts
For training, 4 scripts are provided. All are designed to train models on the dataset generated with the above method.
- `train_gru.py` : Trains GRU model.
- `train_lstm.py` : Trains LSTM model.
- `train_parallel.py` : Trains GPT model on multiple GPUs, using `torch.nn.Dataparallel`
- `train_script.py` : Trains GPT model on a single GPU.


For all 4 scripts, usage is as follows :
```usage: train_script.py [-h] [-d DEVICE] [-t TOKENIZER_PATH] [-p PROJECT_NAME] [-s] file_location

Starts training of Predictor model given a JSON config file.

positional arguments:
  file_location         Path to the JSON config file. Relative to where you launch the script from.

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        Device string, e.g. 'cuda:0' or 'cpu'
  -t TOKENIZER_PATH, --tokenizer_path TOKENIZER_PATH
                        Path for the tokenizer to use (only used for logging snippets). Relative to the train_script folder.
  -p PROJECT_NAME, --project_name PROJECT_NAME
                        Name of the project to log to. Default is 'BackPerplexityResilient'
  -s, --no_step_pickup  If set, will train for steps_to_train more steps. Otherwise, will train up to steps_to_train steps (picking up where it left off)
  ```

### JSON config file
To run the training script, we need to provide it with a path to the JSON config file. Their format slightly depends if training a GPT, GRU or LSTM model. In a nutshell, they contain all the necessary hyperparameters for a training run.

Here is a description of each entry : 

```
{
    "model_params": { # Model parameters
        "vocab_size": 50257, # Vocabulary size
        "n_layers": 12, # Number of Transformer Blocks
        "n_heads": 12, # Number of attention heads
        "embed_dim": 768, # Number of hidden/embedding dimensions
        "attn_length": 256, # Attention Length
        "mlp_ratio": 4.0, # MLP ratio
        "dropout": 0.1, # Dropout inside tranformer blocks
        "embd_dropout": null # Dropout for the token embeddings. Defaults to 0.
    },
    "training_params": { 
        "dataset_folder": "english/english.h5", # Location of .h5 dataset to train on
        "batch_size": 180, # Batch size
        "aggregate": 1, # Number of times to aggregate gradients before gradient step. (effective batch_size = aggregate*batch_size)
        "backwards": false, # Whether to train in the backwards direction
        "steps_to_train": null, # Number of gradient steps to train. Defaults to one epoch of the dataset.
        "save_every": 3000, # Number of steps between each save of the training state.
        "backup_every": 15000, # Number of steps between a backup of the training state.
        "step_log": 400, # Number of steps between each log of training loss in wandb
        "valid_steps": 1000, # Number of batches seen during one validation.
        "state_save_loc": "datavol/vassilis/runs" # folder in which to save the training state.
    },
    "optim_params": {
        "lr": 0.0001, # Base learning rate
        "warmup_steps": 4000, # Number of batches until learning rate warms up
        "oscil_steps": 300000, # Number of steps between warm restarts
        "lr_shrink": 0.85, # Shrinking factor of lr between warm restarts
        "lr_init": 1e-07, # Initial learning rate, for warmup
        "lr_min": 1e-06 # Minimum learning rate reached during cosine annealing.
    }
}
```