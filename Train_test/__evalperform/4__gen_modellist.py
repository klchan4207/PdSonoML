import os
import re 
import pandas as pd

currentdir = os.getcwd()
#currentdir = os.path.dirname(os.path.realpath(__file__))

modelINFO_updated_df = pd.read_csv(currentdir+'/modelINFO_updated.csv')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Top10', type=str)
parser.add_argument('--Top1', type=str)
parser.add_argument('--vali_loss', type=float)
parser.add_argument('--valid_R_square', type=float)
parser.add_argument('--train_R_square', type=float)
_Top10 = parser.parse_args().Top10
_Top1 = parser.parse_args().Top1
_vali_loss = parser.parse_args().vali_loss
_valid_R_square = parser.parse_args().valid_R_square
_train_R_square = parser.parse_args().train_R_square
if _Top10 == "1":
    pass
elif _Top1 == "1":
    pass
else:
    if _vali_loss==None:
        _vali_loss = 1.5
    if _valid_R_square==None:
        _valid_R_square = 0.8
    if _train_R_square==None:
        _train_R_square = 0.0
    modelINFO_updated_df = modelINFO_updated_df.loc[modelINFO_updated_df['vali_loss'] < _vali_loss]
    modelINFO_updated_df = modelINFO_updated_df.loc[modelINFO_updated_df['valid_R_square'] > _valid_R_square]
    modelINFO_updated_df = modelINFO_updated_df.loc[modelINFO_updated_df['train_R_square'] > _valid_R_square]

with open(currentdir+"/modellist","w+") as output:
    output.write("\n".join(modelINFO_updated_df['model_foldername'].values.tolist()))
