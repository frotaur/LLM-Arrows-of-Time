from .gru import MinGRU
from .gru import MinLSTM
from .gru import MinGRU_Trainer

from .mingpt import MinGPT
from .mingpt import MinGPT_Trainer


def load_model(model_name: str, model_config: dict):
    """
    Loads and return model.

    Args:
        model_name: name of the model to load. 'gru','lstm' or 'gpt'
    """
    model_dict = {"gru": MinGRU, "lstm": MinLSTM, "gpt": MinGPT}
    assert model_name.lower() in [
        "gru",
        "lstm",
        "gpt",
    ], 'Model name should be one of "gru","lstm" or "gpt"'

    return model_dict[model_name.lower()](**model_config)


def load_trainer(model_name: str, trainer_config: dict):
    """
    Loads and return trainer.

    Args:
        model_name: name of the model to load. 'gru','lstm' or 'gpt'
        trainer_config: dict of parameters to the trainer
    """

    trainer_dict = {
        "gru": MinGRU_Trainer,
        "lstm": MinGRU_Trainer,
        "gpt": MinGPT_Trainer,
    }

    assert model_name.lower() in [
        "gru",
        "lstm",
        "gpt",
    ], 'Model name should be one of "gru","lstm" or "gpt"'

    return trainer_dict[model_name.lower()](**trainer_config)
