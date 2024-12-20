"""
Base training function for language models.
"""

import os
import json
import shutil
from pathlib import Path

import numpy as np

import torch
import torch.optim
from torch.utils.data import Subset

from modules import load_model
from modules import load_trainer
from modules import TokenTextBOS as TokenTextBOS
from modules import TokenTextFWBW as TokenTextFWBW
from modules import get_tokenizer

from torchenhanced import CosineWarmup
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm


def train(
    model_name: str,
    file_location: str,
    device: str | list[str],
    project_name: str = None,
    run_name: str = None,
    step_pickup: bool = True,
    cooldown_now: bool = False,
):
    """
    Main train script. Launches training of a model given parameters in a JSON file at file_location

    Args:
        model_name: name of the model to train. Must be a valid model in the modules
        file_location: path to the JSON config file
        device: device to train on. 'cpu' or 'cuda:0'. Can be a list of devices
            for data parallelism.
        project_name: name of the project to log to.
        run_name: name of the specific run. If None, will use the name of the JSON file.
        step_pickup: If false, train steps_to_train steps more. If true, will
            train UP TO steps_to_train TOTAL steps.
        val_batch_size: When creating validation dataset, will do it assuming
        a batch size of val_batch_size.
        cooldown_now: If true, will cool down the learning rate to 0, for 10% of the training time.
    """

    cur_path = Path(__file__).parent.absolute()
    file_location = Path(file_location)
    try:
        with open(file_location, "r") as f:
            configo = json.load(f)
            model_params = configo["model_params"]
            training_params = configo["training_params"]
            optim_params = configo["optim_params"]
    except Exception as e:
        print("Error reading JSON file !")
        raise (e)


    if run_name is None:
        run_name = file_location.stem

    tokenizer_path = training_params.get("tokenizer_path",None)
    tokenizer_path = cur_path / Path(tokenizer_path)
    print("TOKENIZER PATH : ", tokenizer_path)
    if tokenizer_path is None:
        raise (
            ValueError(
                "Tokenizer path required, please specify in the .json file with key 'tokenizer_path'"
            )
        )
    if not tokenizer_path.is_dir():
        raise (
            FileNotFoundError(f"Tokenizer not found at path {tokenizer_path}. \n \
                                Tokenizer path should be relative to train.py directory.")
        )

    tokenizer = get_tokenizer(m_path=tokenizer_path.as_posix())

    if not os.path.exists(training_params["dataset_folder"]):
        raise FileNotFoundError(f"Tried to find dataset folder at \
                                {training_params['dataset_folder']}, but failed. \
                                Make sure there is the folder {training_params['dataset_folder']}\
                                in the same directory.")

    if isinstance(device, (list, tuple)):
        parallel = list(device)
        device = parallel[0]
    else:
        parallel = None

    # We ask how many validation steps. To get these, we assume 4% of training
    # time allocated for validation.
    valid_steps = training_params["valid_steps"]
    valid_percent_time = 4  # Time spent validating, in percentage
    valid_percent_time = valid_percent_time / 100

    valid_every = int(valid_steps / valid_percent_time)

    print(f"Validating every {valid_every} steps")

    backwards = training_params.get("backwards",None)  # If we train backwards
    permutation = training_params.get("permutation",None)  # If we train with a permutation

    rng = np.random.default_rng(42)  # For deterministic shuffling of dataset

    # First copy the dataset in the current folder.
    # This is useful in the case of a network drive, where the dataset is slow to access.
    # Can be removed if dataset location is local.

    dataset_path = training_params["dataset_folder"]
    destination_path = os.path.join(cur_path, "local_dataset.h5")

    if not os.path.exists(destination_path):
        print("Copying dataset to current folder...")
        shutil.copy(dataset_path, destination_path)
    else:
        print("Dataset already copied to current folder, using this one.")

    # Generate and shuffle the dataset
    if((backwards is not None) or (permutation is not None)):
        # If we have a definite direction, we use it
        motherDataset = TokenTextBOS(
            h5_file=destination_path,
            attn_length=model_params["attn_length"],
            backwards=backwards,
            permutation=permutation
        )
    else:
        if(tokenizer.special_tokens()=={}):
            tokenizer.add_special()
        fw_num = tokenizer.special_tokens()['<|forward|>']
        bw_num = tokenizer.special_tokens()['<|backward|>']
        model_params["vocab_size"] = model_params["vocab_size"]+2 # Adjust vocab size

        motherDataset = TokenTextFWBW(h5_file=destination_path,
                                      attn_length=model_params["attn_length"],
                                      fw_token_num=fw_num,
                                      bw_token_num=bw_num,
                                      stride=model_params["attn_length"])

    if training_params["fast_scrambling"]:
        # Optional, to speed up scrambling, and limit
        # the size of the dataset which is shuffled, avoiding
        # memory issues.
        max_indices = len(motherDataset)
        MAX_INDICES = 2 * 10**9  # 3 billion tokens
        bunch_size = 5 * 10**8
        indices = []
        if max_indices > MAX_INDICES:
            # We remove the last one if we have at least 3 billion tokens
            shuffled_indices = rng.choice(
                np.arange(max_indices // bunch_size),
                min(3, max_indices // bunch_size),
                replace=False,
            )
            print(
                "doing fast scrambling, chosen elements : ",
                shuffled_indices,
                "among ",
                max_indices // bunch_size,
            )
        else:
            # We do not remove the last one if we have less than 3 billion tokens
            shuffled_indices = rng.choice(
                np.arange(max_indices // bunch_size + 1),
                min(3, max_indices // bunch_size + 1),
                replace=False,
            )

        for i in tqdm(shuffled_indices):
            # shuffle max 10**9 tokens at a time
            batch_indices = np.arange(
                i * bunch_size, min((i + 1) * bunch_size, max_indices)
            )
            rng.shuffle(batch_indices)
            indices.extend(list(batch_indices))
    else:
        indices = np.arange(len(motherDataset))
        rng.shuffle(indices)

    motherDataset = Subset(motherDataset, list(indices))  # Shuffled dataset

    val_batch_size = training_params.get("val_batch_size", None)
    if val_batch_size is None:
        val_batch_size = training_params["batch_size"]
        print("WARNING : val_batch_size not specified, using batch_size instead.")

    # I take val_batch_size*valid_steps datapoints in the validation set
    val_inds = valid_steps * val_batch_size

    # Validation, last portion of dataset
    assert len(motherDataset) > val_inds, """There is not enough data for validation! This can be fixed either by reducing the number of validation steps, or by reducing the validation batch size."""
    val_range = range(len(motherDataset) - val_inds, len(motherDataset))
    # Training, first portion of dataset
    keep_range = range(len(motherDataset) - val_inds)

    del indices

    # Whether backwards or forwards, its the individual examples that are flipped, not the dataset. So same thing for both !
    train_dataset = Subset(motherDataset, keep_range)
    val_dataset = Subset(motherDataset, val_range)

    print("Datapoint example. Check it looks correct ! : ")
    print("TRAIN DATA   :", tokenizer.detokenize(train_dataset[0][0][:20]))
    print("TRAIN ANSWER : ", tokenizer.detokenize(train_dataset[0][1][:20]))
    print("VALID DATA   :", tokenizer.detokenize(val_dataset[0][0][:20]))
    print("VALID ANSWER : ", tokenizer.detokenize(val_dataset[0][1][:20]))

    model = load_model(model_name, model_params)

    # ====================== TRAINING PARAMETERS =======================
    batch_size = training_params["batch_size"]
    aggregate = training_params["aggregate"]
    totbatches = len(train_dataset) // batch_size

    if training_params["steps_to_train"] is None:
        steps_to_train = totbatches
    else:
        steps_to_train = training_params["steps_to_train"]

    print(f"{totbatches=}, {batch_size=}, {len(train_dataset)=}")
    base_lr = optim_params["lr"]


    print(f"--- Training for ~ {steps_to_train//1000}k minibatches ---")
    # ------ Optimizers ------
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr)

    # ------ Schedulers ------
    warmup_steps = optim_params["warmup_steps"]
    
    # Constant LR, with linear warmup
    scheduler = LinearLR(optim, start_factor=1e-7, end_factor=1, total_iters=warmup_steps)

    # Intialize the trainer class
    trainer_config = dict(
        model=model,
        optim=optim,
        scheduler=scheduler,
        train_dataset=train_dataset,
        valid_dataset=val_dataset,
        detokenizer=tokenizer,
        run_name=run_name,
        project_name=project_name,
        save_loc=training_params["save_loc"],
        backwards=backwards,
        device=device,
        parallel=parallel,
        run_config={
            "model_params": model_params,
            "train": training_params,
            "opti": optim_params,
        },
    )

    trainer = load_trainer(model_name, trainer_config=trainer_config)

    if os.path.exists(
        os.path.join(
            training_params["save_loc"],
            project_name,
            "state",
            run_name + ".state",
        )
    ):
        trainer.load_state(
            os.path.join(
                training_params["save_loc"],
                project_name,
                "state",
                run_name + ".state",
            )
        )

    trainer.stepnum = 1
    trainer.train_steps(
        steps=steps_to_train,
        save_every=2000,
        aggregate=aggregate,
        backup_every=training_params["backup_every"],
        step_log=training_params["step_log"],
        batch_size=batch_size,
        valid_every=valid_every,
        resume_batches=True,
        pickup=step_pickup,
        cooldown_now=cooldown_now,
        cooldown_finish=training_params["cooldown_finish"]
    )
