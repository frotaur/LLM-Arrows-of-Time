import json,os


param_dir = 'TrainParams/XL/'
def create_languages(language_list):
    base_forward = {}
    base_backward = {}

    base_path = 'datavol/vassilis/'
    with open (os.path.join(param_dir,'english_XL.json')) as f:
        base_forward = json.load(f)
    with open (os.path.join(param_dir,'english_XL_b.json')) as f:
        base_backward = json.load(f)
    
    for language in language_list:
        new_forward = base_forward.copy()
        new_forward['training_params']['dataset_folder'] = f'{base_path}{language}/{language}.h5'
        new_backward = base_backward.copy()
        new_backward['training_params']['dataset_folder'] =f'{base_path}{language}/{language}.h5'

        
        with open(os.path.join(param_dir,language+'_XL.json'),'w') as f:
            json.dump(new_forward,f,indent=4)
        with open(os.path.join(param_dir,language+'_XL_b.json'),'w') as f:
            json.dump(new_backward,f,indent=4)

langlist = ['deutsch', 'french', 'finnish', 'hellas', 'viet', 'turkish','indo']

create_languages(langlist)