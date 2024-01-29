import json, os

def modifparam():
    # Load the original JSON data
    for file in os.listdir('.') :
        if file.endswith('.json'):
            with open(file, 'r') as f:
                data = json.load(f)
                data['training_params']['dataset_folder'] = "datavol/vassilis/english/englisho_0.h5"
                with open(file, 'w') as f:
                    json.dump(data, f, indent=4)

    return "JSON files created successfully."

# Example usage
print(modifparam())