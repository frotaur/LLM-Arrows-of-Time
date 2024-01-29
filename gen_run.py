import json,os
from pathlib import Path
curfold = Path(__file__).parent.as_posix()
from modules.tokenizer import get_tokenizer

type_dict = dict(
    vocab_size=int,n_layers=int,
    n_heads=int,embed_dim=int,
    attn_length=int,mlp_ratio=float,
    dropout=float,embd_dropout=float,
    batch_size=int,aggregate=int,
    dataset_folder=str, backwards=bool,
    steps_to_train=int,backup_every=int,
    step_log=int, valid_steps=int,lr=float,
    warmup_steps=int,oscil_steps=int,
    lr_shrink=float,lr_init=float,
    lr_min=float, state_save_loc=str,
    save_every=int
)

def get_input_for_dict(param_dict):
    for key, default_value in param_dict.items():
        # Get type of the default value to cast the input to the correct type
        value_type = type_dict[key]
        
        # Get input and convert it to the correct type
        if value_type is bool:
            user_input = input(f"Enter value for {key} (t/f) ({default_value} by default): ")
            if user_input.lower() in ["true", "t", "1", "yes", "y"]:
                param_dict[key] = True
            elif user_input.lower() in ["false", "f", "0", "no", "n"]:
                param_dict[key] = False
        else:
            user_input = input(f"Enter value for {key} ({default_value} by default): ")
            if user_input.lower() == 'none':
                param_dict[key] = None
            else :
                try:
                    param_dict[key] = value_type(user_input) if user_input else default_value
                except ValueError:
                    if(value_type==int):
                        param_dict[key] = value_type(float(user_input))

    print(f'will serialize : {param_dict}')
    return param_dict

if __name__ == "__main__":
    tokenizer = get_tokenizer(m_name='gpt2')


    model_params = dict(
        vocab_size=tokenizer.vs,
        n_layers=12,
        n_heads=12,
        embed_dim=768,
        attn_length=128,
        mlp_ratio=4.,
        dropout=0.1,
        embd_dropout=None
    )

    training_params = dict(
        dataset_folder=None,
        batch_size=128,
        aggregate=2,
        backwards=False,
        steps_to_train=None,
        save_every=2500,
        backup_every=15000,
        step_log=250, 
        valid_steps=1000,
        state_save_loc="datavol/vassilis/runs"
    )

    optim_params = dict(
        lr=5e-4,
        warmup_steps=2000,
        oscil_steps=int(3e4),
        lr_shrink=0.95,
        lr_init=1e-7,
        lr_min=1e-6
    )

    print("Enter values for model_params:")
    model_params = get_input_for_dict(model_params)

    print("\nEnter values for training_params:")
    training_params = get_input_for_dict(training_params)

    print("\nEnter values for optim_params:")
    optim_params = get_input_for_dict(optim_params)

    all_params = {
        "model_params": model_params,
        "training_params": training_params,
        "optim_params": optim_params
    }
    user_input = input(f"Enter name for paramset : ")
    name = str(user_input) if user_input else 'params'
    os.makedirs(os.path.join(curfold,'TrainParams'),exist_ok=True)
    with open(os.path.join(curfold,'TrainParams',name+".json"), "w") as file:
        json.dump(all_params, file, indent=4)

    print(f"\nParameters saved to {name}.json")