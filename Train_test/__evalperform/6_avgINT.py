import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
currentdir = os.getcwd()
#currentdir = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modellist', type=str)
parser.add_argument('--include_CONT', type=int)
parser.add_argument('--mode', type=str)

_include_CONT = parser.parse_args().include_CONT
_mode = parser.parse_args().mode
_modellist = parser.parse_args().modellist

if _modellist == None:
    model_folder_list = [model_folder for model_folder in os.listdir(currentdir) 
                                if "_model" in model_folder and model_folder[0]!="~"
                                and os.path.isdir(os.path.join(currentdir, model_folder))]
else:
    with open(currentdir+'/'+_modellist,"r") as input:
        model_folder_list = input.read().split("\n")
        model_folder_list = [_x for _x in model_folder_list if _x != ""]
    
model_folder_list = [_x for _x in model_folder_list
                        if _x[0]!="~"
                        ]

if not _include_CONT:
    model_folder_list = [_x for _x in model_folder_list
                            if "_CONT" not in _x
                            ]

_output_dict = {
    'inputs_E1_Weighted':[],
    'inputs_E2_Weighted':[],
    'y_pred':[],
    'y_true':[],
}

datapt_totalnum = -1

for model_folder in model_folder_list:
    _df = pd.read_csv(currentdir+"/"+model_folder+"/_INT_dict.csv")

    datapt_totalnum = len(_df.values)

    for v, _ in _output_dict.items():
        _output_dict[v].append(_df[v].values.tolist())
    
from statistics import mean
for v,k in _output_dict.items():
    _data = np.array(k).T.tolist()
    if len(_data) != datapt_totalnum:
        print("ERROR in length")
        exit()
    _output_dict[v] = [mean(_x) for _x in _data]


for _col in ['data_type','tag','rct_id','lig_id']:
    _output_dict[_col] = _df[_col].values.tolist()

ouput_df = pd.DataFrame.from_dict(_output_dict)

ouput_df.to_csv("avgINT.csv",index=None)