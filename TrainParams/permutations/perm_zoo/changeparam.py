import json, os
import sys

sys.path.append("../../../..")
from perm_generator import *

perm_list = [
    forward,
    backwards,
    middle_out,
    end_in,
    even_odd,
    odd_even,
    half_back_forward,
    half_forward_back,
    quarter_middle_out_interleave,
    quarter_middle_out_concat,
    quarter_end_in_interleave,
    quarter_end_in_concat,
    lambda att : random_chunk(att,0),
    lambda att : random_chunk(att,5),
    lambda att : uniform_random(att,0),
    lambda att : uniform_random(att,5),
    lambda att : uniform_random(att,10),
    lambda att : uniform_random(att,15)
]

perm_names = [
    "fw",
    "bw",
    "mid_out",
    "end_in",
    "even_odd",
    "odd_even",
    "half_bwff",
    "half_fwbw",
    "qmid_out_int",
    "qmid_out_cat",
    "qend_in_int",
    "qend_in_cat",
    "rand_chunk1",
    "rand_chunk2",
    "uniform_rand1",
    "uniform_rand2",
    "uniform_rand3",
    "uniform_rand4",
]


def modifparam():
    att_size = 256
    # Load the original JSON data
    with open("fr_256_blueprint.json", "r") as f:
        data = json.load(f)
        for permutation, perm_name in zip(perm_list, perm_names):
            data["training_params"]["permutation"] = permutation(att_size)
            with open("fr_256_" + perm_name + ".json", "w") as f:
                json.dump(data, f, indent=4)

    return "JSON files created successfully."

def modif_steps():
    for file in os.listdir():
        if file.endswith(".json"):
            with open(file, "r") as f:
                data = json.load(f)
                data["training_params"]["steps_to_train"] = 400000
                data["training_params"]["cooldown_finish"] = 0.15
            with open(file, "w") as f:
                json.dump(data, f, indent=4)

modif_steps()