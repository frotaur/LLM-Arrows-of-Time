"""
    Training script for GPT-like models. Use this along with the output of gen_run !
"""
import sys

sys.path.append('../../../')
print(sys.path)
from modules import *
import os,json, pathlib


cur_path = pathlib.Path(__file__).parent.absolute().as_posix()


def getparanum(file_location, device='cpu') :

    with open(file_location,'r') as f :
        configo = json.load(f)
        model_params = configo['model_params']


    model = MinGPT(**model_params)
    print('TOTAL PARANUMBER ', model.paranum)

if __name__=='__main__':
    files = [path for path in os.listdir(cur_path) if path.endswith('.json') ]

    for file in files :
        print(f'=================={os.path.basename(file)}==================')
        getparanum(file, device='cpu')


    

