import os
import re 
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Top10', type=str)
parser.add_argument('--Top1', type=str)
parser.add_argument('--print_zip', type=str)
_Top10 = parser.parse_args().Top10
_Top1 = parser.parse_args().Top1
_print_zip = parser.parse_args().print_zip

currentdir = os.getcwd()

with open(currentdir+'/__DONEmodels_loss_log','r') as input:
    _txt = input.read()

modelINFO_dict = {
                    'model_foldername':[],
                    'study':[],
                    'loss':[],
                    'vali_loss':[],
                    'test_loss':[],
                }
# separate all models
model_readlines = re.split('\n\d+\n',_txt[2:])
for readlines in model_readlines:
    _count = 1
    readlines_trt = [_x for _x in readlines.split('\n') if _x != '']
    # check format of info
    if len(readlines_trt) != 5:
        print(readlines_trt)
        print('ERROR while splitting')
        exit()
    # store loss info one by one
    for lines in readlines_trt:
        if _count==1:
            # model_foldername
            modelINFO_dict['model_foldername'].append(lines)
        else:
            # study loss vali_loss test_loss
            k,v = lines.split(' ')
            if v=='':
                v = 0
            modelINFO_dict[k].append(float(v))
        _count+=1
# turn to dataframe format
modelINFO_df = pd.DataFrame.from_dict(modelINFO_dict)
# sort model based on validation loss
modelINFO_df = modelINFO_df.sort_values(['vali_loss'], ascending = (True))

if _Top10 == '1':
    # just give out TOP 10 model based on validation loss
    modelINFO_df = modelINFO_df.head(10)
elif _Top1 == '1':
    # just give out TOP 10 model based on validation loss
    modelINFO_df = modelINFO_df.head(1)
else:
    # filter based on validation loss
    modelINFO_df = modelINFO_df.loc[modelINFO_df['vali_loss'] < 2.0]

if _print_zip == '1':
    filtered_model_list = modelINFO_df['model_foldername'].values.tolist()
    print("zip -r modelpkg.zip "+" ".join(filtered_model_list))

# store as .csv
modelINFO_df.to_csv(currentdir+'/modelINFO.csv',index=False)