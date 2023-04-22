import pandas as pd
import os
import json

currentdir = os.getcwd()
#currentdir = os.path.dirname(os.path.realpath(__file__))

modelINFO_df = pd.read_csv(currentdir+'/modelINFO.csv')

model_foldername_list = modelINFO_df['model_foldername'].values.tolist()

update_dict = {'model_foldername':[]}
for model_foldername in model_foldername_list:
    print(model_foldername)
    update_dict['model_foldername'].append(model_foldername)
    with open(currentdir+"/"+model_foldername+"/"+"_eval_result","r") as input:
        # performance dict
        perf_dict = json.loads(input.read())
        for _datatype,_innerdict in perf_dict.items():
            for k,v in _innerdict.items():
                update_colname =  _datatype+"_"+k
                if update_dict.get(update_colname):
                    update_dict[ update_colname ].append(v)
                else:
                    update_dict[ update_colname ] = [v]

update_df = pd.DataFrame.from_dict(update_dict)

modelINFO_df = pd.concat([modelINFO_df, update_df], axis=1)

modelINFO_df.to_csv(currentdir+'/modelINFO_updated.csv',index=False)