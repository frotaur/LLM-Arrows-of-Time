<div align="center">

<h2>Arrows of Time for Large Language Models</h2> 

  <a href='https://arxiv.org/abs/2401.17505'><img src='https://img.shields.io/badge/ArXiv-2401.17505-red'></a> 

  <div>
      <a href='https://scholar.google.com/citations?user=4o52I2oAAAAJ&hl=en' target='_blank'>Vassilis Papadopoulos<sup>* 1 2</sup> </a>&emsp;
      <a href='https://jeremiewenger.com/' target='_blank'>Jérémie Wenger<sup>3</a>&emsp;
      <a href='https://scholar.google.com/citations?user=p9B6eWEAAAAJ&hl=en'_blank'>Clément Hongler<sup>* 2</sup></a>&emsp;
  </div>
  <br>
  <div style='font-size: 8pt'>
      <sup>1</sup> FSL/Institute of Physics, EPFL, Switzerland &emsp; <sup>2</sup> CSFT/Institute of Mathematics, EPFL, Switzerland &emsp; <sup>3</sup> Department of Computing, Goldsmiths/UoL, London, UK
  </div>
  <div style='font-size: 8pt'>Correspondence to: Clément Hongler, clement.hongler©epfl.ch</div>
  <div style='font-size: 8pt'><sup>*</sup> equal contributions</div>

  <br>

  <h3>Abstract</h3>

  <div style='text-align:justify'>We study the probabilistic modeling performed by Autoregressive Large Language Models (LLMs) through the angle of time directionality, addressing a question first raised in (Shannon, 1951). For large enough models, we empirically find a time asymmetry in their ability to learn natural lan- guage: a difference in the average log-perplexity when trying to predict the next token versus when trying to predict the previous one. This difference is at the same time subtle and very consistent across various modalities (language, model size, training time, ...). Theoretically, this is surprising: from an information-theoretic point of view, there should be no such difference. We provide a theoretical framework to explain how such an asymmetry can appear from sparsity and compu- tational complexity considerations, and outline a number of perspectives opened by our results.</div>

  <br>

  <figure>
    <img src='pics/en-fr.train-valid.combined_with_inset.svg'>
    <legend>English vs French validation losses (French training losses in the zoom-in, early loss values cropped for readability). </legend>
  </figure>

</div>



---

## Installation

Install requirements using `pip install -r requirements.txt`.

NOTE : On Windows, doing this might install torch without CUDA support. If this is the case, first install pytorch CUDA following instruction on the official [website](https://pytorch.org/), then run `pip install -r requirements.txt`.

Read the following section to learn how to reproduce experiments.

## Tokenization
The script `tokenize_to_h5.py` can be used to prepare a dataset for training. Given a .txt file, it will train a BPE tokenizer on it, then use it to tokenize the text, and save the tokenized dataset in `.h5` format.

CC100 datasets can be downloaded [here](https://data.statmt.org/cc-100/). 

### Usage :

To use `tokenize_to_h5.py`, first put a standalone `.txt` file inside a folder. Then, use `tokenize_to_h5.py` using the following arguments
``` 
usage: tokenize_to_h5.py [-h] --txt_path TXT_PATH

        Script for preparing a .txt cc-100 dataset for training. Creates the
        custom tokenizer, and tokenizes the text with it to generate the .h5
        file for training.

        To make one of those things independently (e.g., only make the custom
        tokenizer), see modules/tok_utils.
        

options:
  -h, --help            show this help message and exit
  --txt_path TXT_PATH, -t TXT_PATH
                        
                                The input file to be tokenized. This script will save the following
                                items:
                                1) given the path of a source plain text file, a folder of the same
                                name as the containing folder of txt_path, with '_h5' appended at the
                                end, as well as raw Pytorch tensors. Example:
                                    -t my_dataset/input.txt -> my_dataset_h5/input.h5
                                                               my_dataset_pt/input_tokenized.pt
                                2) a tokenizer in modules/tokenizers called after the folder containing
                                the txt dataset. Example:
                                    -t code_dataset/input.txt -> modules/tokenizers/code_dataset_tokenizer/
```

Then run the script.

NOTE : tokenization of large .txt files (>100GB) might take a while (1,2 days). This script is NOT designed to pick up where it left off if it crashes. For bigger datasets, consider making a script (include `from modules.tok_utils import *`), and run, subsequently :  
- `create_tokenizer(txt_path, tokenizer_folder,tokenizer_name=tokenizer_name)` : Will train the BPE tokenizer on the given .txt file, and save it in <tokenizer_folder>/<tokenizer_name>  
- `tokenize_folder(os.path.dirname(txt_path), os.path.join(tokenizer_folder,tokenizer_name))` : Will tokenize the text file, splitting it into subfiles if necessary for memory reasons. Saved the tokenized tensors as `.pt`. If it crashes mid-way, can be restarted, and will pickup from last checkpoint  
- `make_h5(os.path.dirname(txt_path), os.path.splitext(os.path.basename(txt_path))[0], out_h5_folder,toki)` : Will convert a folder containing `.pt` files into a single `.h5` dataset, ready for training.

For more informations on these functions, look at docstring comments in `modules/tok_utils`

### Tokenizer class

The tokenizer class we use throughout the project is defined in `modules/tokenizer.py`. It is a wrapper on top of the Huggingface tokenizer.

Here is all you need to know to use the tokenizers :

```python
from modules import tokenizer

toki = tokenizer.get_tokenizer(m_path='modules/tokenizers/en_tokenizer') # Load a saved tokenizer by specifying saved folder
# A saved tokenizer is created by using create_tokenizer in modules/tok_utils/create_custom_tokenizer.py

tokenized = toki.tokenize("Hello, world!") # Tokenize a string
print(tokenized) # Get a tensor of ints shape [1, seq_len]
print(toki.detokenize(tokenized)) # Detokenize a tensor of ints, prints "Hello, world!"
```

Note: the scripts in `modules/tok_utils/` can, to some degree, be run independently, provided the path to the module folder is added to the PYTHONPATH: `PYTHONPATH="/path/to/modules:$PYTHONPATH" python modules/tok_utils/pt_to_h5.py --help`.

## Training

### Scripts
For training, 4 scripts are provided. All are designed to train models on the dataset generated with the above method.
- `train_gpt.py` : Trains GPT model on a single GPU.
- `train_gru.py` : Trains GRU model. 
- `train_lstm.py` : Trains LSTM model.
- `train_parallel.py` : Trains GPT model on multiple GPUs, using `torch.nn.Dataparallel`


For all 4 scripts, usage is as follows :
```
usage: train_xxx.py [-h] [-d DEVICE] [-t TOKENIZER_PATH] [-p PROJECT_NAME] [-s] file_location

Starts training of Predictor model given a JSON config file.

positional arguments:
  file_location         Path to the JSON config file. Relative to where you launch the script from.

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                Device string, e.g. 'cuda:0' or 'cpu'. For parallel, list of devices.
  -t TOKENIZER_PATH, --tokenizer_path TOKENIZER_PATH
                Path for the tokenizer to use (only used for logging snippets). Relative to the script folder.
  -p PROJECT_NAME, --project_name PROJECT_NAME
                Name of the project to log to. 
  -s, --no_step_pickup 
                If set, train steps_to_train steps more. Otherwise, will train UP TO steps_to_train TOTAL steps."
  ```

Example :
```bash
python train_script.py path/to/config.json -d cuda:0 -t path/to/tokenizer -p MyTrainingProject -s
```

### JSON config file
To run the training script, we need to provide it with a path to the JSON config file. Their format slightly depends if training a GPT, GRU or LSTM model. In a nutshell, they contain all the necessary hyperparameters for a training run.

Here is a description of each entry : 

```json
{
    "model_params": {                              # Model parameters
        "vocab_size": 50257,                       # Vocabulary size
        "n_layers": 12,                            # Number of Transformer Blocks
        "n_heads": 12,                             # Number of attention heads
        "embed_dim": 768,                          # Number of hidden/embedding dimensions
        "attn_length": 256,                        # Attention Length
        "mlp_ratio": 4.0,                          # MLP ratio
        "dropout": 0.1,                            # Dropout inside tranformer blocks
        "embd_dropout": null                       # Dropout for the token embeddings. Defaults to 0.
    },
    "training_params": { 
        "dataset_folder": "english/english.h5",    # Location of .h5 dataset to train on
        "batch_size": 180,                         # Batch size
        "aggregate": 1,                            # Number of times to aggregate gradients before gradient step. (effective batch_size = aggregate*batch_size)
        "backwards": false,                        # Whether to train in the backwards direction
        "steps_to_train": null,                    # Number of gradient steps to train. Defaults to one epoch of the dataset.
        "save_every": 3000,                        # Number of steps between each save of the training state.
        "backup_every": 15000,                     # Number of steps between a backup of the training state.
        "step_log": 400,                           # Number of steps between each log of training loss in wandb
        "valid_steps": 1000,                       # Number of batches seen during one validation.
        "state_save_loc": "datavol/vassilis/runs"  # folder in which to save the training state.
    },
    "optim_params": {
        "lr": 0.0001,                              # Base learning rate
        "warmup_steps": 4000,                      # Number of batches until learning rate warms up
        "oscil_steps": 300000,                     # Number of steps between warm restarts
        "lr_shrink": 0.85,                         # Shrinking factor of lr between warm restarts
        "lr_init": 1e-07,                          # Initial learning rate, for warmup
        "lr_min": 1e-06                            # Minimum learning rate reached during cosine annealing.
    }
}
```
