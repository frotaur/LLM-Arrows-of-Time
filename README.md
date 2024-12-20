<div align="center">

<h2>Arrows of Time for Large Language Models</h2> 

  <a href='https://arxiv.org/abs/2401.17505'><img src='https://img.shields.io/badge/ArXiv-2401.17505-red'></a> 

  <div>
      <a href='https://scholar.google.com/citations?user=4o52I2oAAAAJ&hl=en' target='_blank'>Vassilis Papadopoulos<sup>* 1 2</sup> </a>&emsp;
      <a href='https://jeremiewenger.com/' target='_blank'>Jérémie Wenger<sup>3</a>&emsp;
      <a href='https://scholar.google.com/citations?user=p9B6eWEAAAAJ&hl=en'_blank'>Clément Hongler<sup>* 1</sup></a>&emsp;
  </div>
  <br>
  <div style='font-size: 8pt'>
      <sup>1</sup> CSFT/Institute of Mathematics, EPFL, Switzerland &emsp; <sup>2</sup> FSL/Institute of Physics, EPFL, Switzerland &emsp;  <sup>3</sup> Department of Computing, Goldsmiths/UoL, London, UK
  </div>
  <div style='font-size: 8pt'>Correspondence to: Clément Hongler, clement.hongler©epfl.ch</div>
  <div style='font-size: 8pt'><sup>*</sup> equal contributions</div>

  <br>

  <h3>Abstract</h3>

  <div style='text-align:justify'>We study the probabilistic modeling performed by Autoregressive Large Language Models (LLMs) through the angle of time directionality, addressing a question first raised in (Shannon, 1951). For large enough models, we empirically find a time asymmetry in their ability to learn natural lan- guage: a difference in the average log-perplexity when trying to predict the next token versus when trying to predict the previous one. This difference is at the same time subtle and very consistent across various modalities (language, model size, training time, ...). Theoretically, this is surprising: from an information-theoretic point of view, there should be no such difference. We provide a theoretical framework to explain how such an asymmetry can appear from sparsity and compu- tational complexity considerations, and outline a number of perspectives opened by our results.</div>

  <br>

  <figure>
    <img src='pics/all-med-lang.png'>
    <legend> Losses during training for different languages, trained forwards and backwards. For all languages, the backward model does slightly worse! </legend>
  </figure>

</div>



---

## Installation

Install requirements using `pip install -r requirements.txt`.

NOTE : On Windows, doing this might install torch without CUDA support. If this is the case, first install pytorch CUDA following instruction on the official [website](https://pytorch.org/), then run `pip install -r requirements.txt`.

Read the following section to learn how to reproduce experiments.

## Branches and project versions
We recommend to always use the latest commit on the 'main' branch, as it will always be the cleanest and most updated branch. Tags are present on the git tree to restore the project to previous versions.

### Branch : main
Main branch, may be regularly updated to increase code readability/usability. For instance, cosine decay of the LR was removed in favor of a linear cooldown at the end of training, which is more convenient (see [here](https://arxiv.org/pdf/2405.18392)). Use this branch to run similar experiments to the natural language ones in the paper. Will contain several examples of `.json`files for training, to get the exact specifications of previous experiments see the tags below.

### Tag : original
This tag restores the codebase to a snapshot as it was for the first submission of the paper. Uses an earlier version of torchenhanced (custom library used for training, akin to pytorch lightning). Contains the `.json` files for all the natural language experiments

### Tag : rebuttal
This tag restores the codebase to a snapshot as it was for the submission of the rebuttal, during the paper review. Contains slightly update code, as well as additional `.json` files for experiments on Arabic, Hebrew and Tagalog, as well reversed French

## Tokenization
The script `tokenization_pipeline.py` can be used to prepare a dataset for training. Given a folder containing .txt files, it will train a BPE tokenizer on them, then use it to tokenize the text, and save the tokenized dataset in `.h5` format. The pipeline uses the huggingface implementation of BPE tokenizers.

CC100 datasets can be downloaded [here](https://data.statmt.org/cc-100/). 

### Quick test :
We provide the folder `shake_test` to test the tokenization pipeline. Simply run `python tokenization_pipeline.py shake_test`. It should generate a folder `shake_test_h5` which contains the tokenized data, as well as `modules/tokenizers/shake_test_tokenizer`, which is the BPE tokenizer that was just trained.
### Usage :

To use `tokenization_pipeline.py`, first put the `.txt` comprising the dataset in a folder. Then, use `tokenization_pipeline.py` using the following argument
``` 
usage: tokenization_pipeline.py [-h] txt_path

        Script for preparing a .txt cc-100 dataset for training. Creates the custom tokenizer, and tokenizes the text with it to generate the .h5 file for training.

        To make one of those things independently (e.g., only make the custom tokenizer), see tokenization_scripts.
        

positional arguments:
  txt_path    
        The input folder to be tokenized. This script will save the following items:
        1) A folder named '<txt_path>_pt', containing the tokenized data as pytorch tensors. A folder named '<txt_path>_h5' containing the tokenized h5py dataset. Example:
            my_dataset/input.txt -> my_dataset_h5/input.h5
                                    my_dataset_pt/input_tokenized.pt
        2) a tokenizer in modules/tokenizers named '<txt_path>_tokenizer'. Example:  
        code_dataset/input.txt -> modules/tokenizers/code_dataset_tokenizer/
                      

options:
  -h, --help  show this help message and exit
```

Then run the script.

NOTE : tokenization of large .txt files (>100GB) might take a while (1,2 days). This script is NOT designed to pick up where it left off if it crashes. For bigger datasets consider using the scripts in `tokenization_scripts/`:  
- `create_tokenizer.py` : Will train the BPE tokenizer on the given .txt file, and save it. See `python create_tokenizer.py --help`
- `tokenize_txt.py` : Will tokenize the text files, splitting it into subfiles if necessary for memory reasons. Saves the tokenized tensors as `.pt`. If it crashes mid-way, can be restarted, and will pickup from last checkpoint. See `python tokenize_txt.py --help`.
- `tensor_to_h5 : Will convert a folder containing `.pt` files into a single `.h5` dataset, ready for training. See `python tensor_to_h5.py --help`.

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

### Quick Test
After generating the shakespear dataset (see _Quick Test_ in the tokenization section), you can run `python train_gpt.py TrainParams/blueprint.json -d <device>`. This should launch the training, logging using wandb. You may be need to be logged into wandb.
### Scripts
For training, 4 scripts are provided. All are designed to train models on the dataset generated with the above method.
- `train_gpt.py` : Trains GPT model on a single GPU.
- `train_gru.py` : Trains GRU model. 
- `train_lstm.py` : Trains LSTM model.
- `train_parallel.py` : Trains GPT model on multiple GPUs, using `torch.nn.Dataparallel`


For all 4 scripts, usage is as follows :
```
usage: train_xxx.py [-h] [-d DEVICE] [-p PROJECT_NAME] [-s] [-c] file_location

Starts training of Predictor model given a JSON config file.

positional arguments:
  file_location         Path to the JSON config file. Relative to where you launch the script from.

options:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                Device string, e.g. 'cuda:0' or 'cpu'. For parallel, list of devices.
  -p PROJECT_NAME, --project_name PROJECT_NAME
                Name of the project to log to. 
  -r RUN_NAME, --run_name RUN_NAME
                Name of the run. Defaults to the '.json' filename
  -s, --no_step_pickup 
                If set, train steps_to_train steps more. Otherwise, will train UP TO steps_to_train TOTAL steps."
  -c, --cooldown_now
                If set, start  cooldown of learning rate immediately.
  ```

Example :
```bash
python train_script.py path/to/config.json -d cuda:0 -p MyTrainingProject -s
```

### JSON config file
To run the training script, we need to provide it with a path to the JSON config file. Their format slightly depends if training a GPT, GRU or LSTM model. In a nutshell, they contain all the necessary hyperparameters for a training run.

We provide the file `TrainParams\blueprint.json`, as a fully formatted json for a training run. To design other experiments, simply modify this file.

Here is a description of each entry : 

```json
{
    "model_params": {
        "vocab_size": 50257, // Vocabulary size. Should match the vocab size of the tokenizer used.
        "n_layers": 24, // Number of transformer layers.
        "n_heads": 16, // Number of attention heads.
        "embed_dim": 1024, // Dimension of the embeddings.
        "attn_length": 256, // Length of the attention window.
        "mlp_ratio": 4.0, // Multiplier for the hidden dimension of the MLP.
        "dropout": 0.1, // Dropout rate.
        "embd_dropout": null // Dropout rate for the position embeddings.
    },
    "training_params": {
        "dataset_folder": "datavol/french/french.h5", // Location of h5 folder generated with the tokenization pipeline
        "tokenizer_path": "modules/tokenizers/fr_tokenizer", // Path to the tokenizer used to generate the dataset. Only used for logging purposes.
        "batch_size": 90, // Batch size for training
        "aggregate": 2, // Number of batches to aggregate before backpropagating
        "backwards": false, // If true, train the model in reverse.
        "permutation" : null,// Optional; a list containing a permutation of attn_length integers. If provided, ignore the 'backwards' parameter, and instead use this permutation to shuffle the input sequence. 
        "steps_to_train": null, // Number of steps to train. If null, train for one epoch.
        "save_every": 3000, // Save the model every <save_every> steps. Overwrites the previous checkpoint.
        "backup_every": 15000, // Save a backup of the model every <backup_every> steps. Does not overwrite the previous checkpoint.
        "step_log": 200, // Log training progress every <step_log> steps.
        "valid_steps": 1000, // Size of the validation set, in batches. 
        "val_batch_size": 200, // Validation batch size, optional. If not given, will be the same as the training batch size. Useful to keep the validation dataset exactly the same, despite different training batch size.
        "save_loc": "datavol/refinements/runs/",  // Location to save the training state. Used to resume training from a checkpoint.
        "fast_scrambling": false, // Uses imperfect scrambling if true. Useful if dataset is so large that the list of integers cannot be stored in memory.
        "cooldown_finish": 0.15 // Fraction of the training steps dedicated to cooldown. Recommended between 0.1 and 0.2.
    },
    "optim_params": {
        "lr": 0.0001, // Learning rate
        "warmup_steps": 10000, // Number of warmup steps
    }
}
``` 

## Citation
If you find this repository useful in your research, please consider citing our work

```bibtex
@article{papadopoulos2024arrows,
  title={Arrows of Time for Large Language Models},
  author={Papadopoulos, Vassilis and Wenger, Jeremy and Hongler, Clément},
  journal={arXiv:2401.17505},
  year={2024}
}
```