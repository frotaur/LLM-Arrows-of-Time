"""
Base training function for language models.
"""

import os
import json
import shutil
import pathlib

import numpy as np

import torch
import torch.optim
from torch.utils.data import Subset

from modules import load_model
from modules import load_trainer
from modules import TokenTextBOS as TokenTextBOS
from modules import get_tokenizer

from torchenhanced import CosineWarmup

from tqdm import tqdm


def train(
    model_name: str,
    file_location: str,
    device: str | list[str],
    tokenizer_path: str,
    project_name: str = None,
    run_name: str = None,
    step_pickup: bool = True,
    val_batch_size: int = 250,
):
    """
    Main train script. Launches training of a model given parameters in a JSON file at file_location

    Args:
        file_location: path to the JSON config file
        device: device to train on. 'cpu' or 'cuda:0'. Can be a list of devices
            for data parallelism.
        tokenizer_path: path to the tokenizer to use. Relative to the
            train_script folder.
        project_name: name of the project to log to.
        step_pickup: If false, train steps_to_train steps more. If true, will
            train UP TO steps_to_train TOTAL steps.
        val_batch_size: When creating validation dataset, will do it assuming
        a batch size of val_batch_size. This is to have a consistent validation
        set even if using different batch_size when training. NOTE : reduce it
        when testing with a small dataset, otherwise it might comprise the full
        data
    """

    cur_path = pathlib.Path(__file__).parent.absolute().as_posix()

    if run_name is None:
        run_name = os.path.splitext(os.path.basename(file_location))[0]

    print("TOKENIZER PATH : ", os.path.join(cur_path, tokenizer_path))
    if tokenizer_path is None:
        raise (
            ValueError(
                "Tokenizer path required, please specify with -t <tokenizer_path>"
            )
        )
    if not os.path.exists(os.path.join(cur_path, tokenizer_path)):
        raise (
            FileNotFoundError(f"Tokenizer not found at path {os.path.join(cur_path,tokenizer_path)}. \n \
                                Tokenizer path should be relative to train_script.py.")
        )

    tokenizer_path = os.path.join(cur_path, tokenizer_path)
    tokenizer = get_tokenizer(m_path=tokenizer_path)

    try:
        with open(file_location, "r") as f:
            configo = json.load(f)
            model_params = configo["model_params"]
            training_params = configo["training_params"]
            optim_params = configo["optim_params"]
    except Exception as e:
        print("Error reading JSON file !")
        raise (e)

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

    # We ask how many validation steps. To get these, we assume 5% of training
    # time allocated for validation.
    valid_steps = training_params["valid_steps"]
    valid_percent_time = 5  # Time spent validating, in percentage
    valid_percent_time = valid_percent_time / 100

    valid_every = int(valid_steps / valid_percent_time)

    print(f"Validating every {valid_every} steps")

    backwards = training_params["backwards"]  # If we train backwards

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
    motherDataset = TokenTextBOS(
        h5_file=destination_path,
        attn_length=model_params["attn_length"],
        backwards=backwards,
    )

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

    # To keep it constant even if we switch batch_size, I take val_batch_size*valid_steps datapoints in the validation set
    val_inds = valid_steps * val_batch_size
    # Validation, last portion of dataset
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
        steps_to_train = totbatches * 20
    else:
        steps_to_train = training_params["steps_to_train"]

    print(f"{totbatches=}, {batch_size=}, {len(train_dataset)=}")
    base_lr = optim_params["lr"]
    warmup_steps = optim_params["warmup_steps"]
    oscil_steps = optim_params[
        "oscil_steps"
    ]  # Period of cosine restart for oscillation
    lr_shrink_factor = optim_params["lr_shrink"]
    lr_min = optim_params["lr_min"]
    lr_init = optim_params["lr_init"]

    print(f"--- Training for ~ {steps_to_train//1000}k minibatches ---")
    # ------ Optimizers ------
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr)
    # Cosine schedule with Warmup, from torchenhanced package
    scheduler = CosineWarmup(
        optim,
        warmup_steps=warmup_steps,
        lr_shrink=lr_shrink_factor,
        lr_init=lr_init,
        T_0=oscil_steps,
        eta_min=lr_min,
    )

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
    )
