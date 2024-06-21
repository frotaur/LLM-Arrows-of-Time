import json, os

def modifparam(oldname,newname,newvalue):
    # Load the original JSON data

    with open(oldname, 'r') as f:
        data = json.load(f)
        # data['training_params']["dataset_folder"] = "datavol/french/french.h5"
        # data['training_params']["state_save_loc"] = "runs"
        data['model_params']['attn_length'] = newvalue
        with open(newname, 'w') as f:
            json.dump(data, f, indent=4)

    return "JSON files created successfully."

for i in [4,8,24,48,96,192]:
    modifparam("fr_16.json",f"fr_{i}.json",i)
    modifparam("fr_16_b.json",f"fr_{i}_b.json",i)