"""
Training script for GPT-like models. Use this along with the output of gen_run!

Usage:
    python train_gpt.py <path_to_json_config_file> -d <device> -t <tokenizer_path> -p <project_name> -s

Example:
    python train_gpt.py TrainParams/params.json -d cuda:0 -t fr -p ArrowsOfTime -s

Note:
    Uses wandb, so you need to have a wandb account and be logged in.
"""

import argparse

from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts training of Predictor model given a JSON config file."
    )

    parser.add_argument(
        "file_location",
        help="""
        Path to the JSON config file. Relative to where you launch the script
        from.
        """,
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="""
        Device string, e.g. 'cuda:0' or 'cpu'
        """,
    )

    parser.add_argument(
        "--run_name",
        "-r",
        type=str,
        help="""
        Name of the run for wandb logging. Default is the name of the
        config file.
        """,
    )

    parser.add_argument(
        "--project_name",
        "-p",
        default="ArrowsOfTime",
        help="""
        Name of the project to log to. Default is 'ArrowsOfTime'
        """,
    )

    parser.add_argument(
        "--no_step_pickup",
        "-s",
        action="store_false",
        help="""
        If set, train steps_to_train steps more. Otherwise, will train UP TO
        steps_to_train TOTAL steps.
        """,
    )

    parser.add_argument(
        "--cooldown_now",
        "-c",
        action="store_true",
        help="""
        If set, cools down learning rate immediately.
        """,
    )
    args = parser.parse_args()

    train(
        model_name="gpt",
        file_location=args.file_location,
        device=args.device,
        project_name=args.project_name,
        run_name=args.run_name,
        step_pickup=args.no_step_pickup,
        val_batch_size=20, # TODO change to 250 !!
        cooldown_now=args.cooldown_now,
    )
