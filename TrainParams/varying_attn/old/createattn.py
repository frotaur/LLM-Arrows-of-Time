import json

def duplicate_json_with_modified_attn_length(input_integers, original_json='french_med_backwards.json'):
    # Load the original JSON data
    with open(original_json, 'r') as file:
        data = json.load(file)

    # Loop through the input integers and create new JSON files
    for i in input_integers:
        # Update the 'attn_length' value
        data['model_params']['attn_length'] = i

        # Create a new JSON file with the updated data
        new_filename = f'fr_{i}_b.json'
        with open(new_filename, 'w') as new_file:
            json.dump(data, new_file, indent=4)

    return "JSON files created successfully."

# Example usage
input_integers = [16,32,64,128,512,1024]
duplicate_json_with_modified_attn_length(input_integers)