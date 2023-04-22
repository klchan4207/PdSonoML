from re import L
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.python.keras.engine import input_layer
import json
import pandas as pd

import os
currentdir = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model_dirname', type=str)
parser.add_argument('_optuna_filename', type=str)

weighting_rule = {
    "inputs_E1":"dense_output|0",
    "inputs_E2":"dense_output|1",
}

model_dirname = parser.parse_args().model_dirname
_optuna_filename = parser.parse_args()._optuna_filename.replace(".py","")

with open(currentdir+"/"+model_dirname+"/"+model_dirname+"_setting","r") as input:
    trial_setup_dict = json.load(input)

import sys
_optuna_filedir, _optuna_filename = os.path.split(_optuna_filename)
sys.path.insert(1, currentdir)
_optuna = __import__(_optuna_filename)

# gen data
model_dir = currentdir + "/" + model_dirname + "/" + model_dirname
    
# Get train/test data.
_output_dict = _optuna.get_data(          
                        trial_setup_dict = trial_setup_dict,   
                        batch_size = 10000000,
                        test_split = trial_setup_dict['test_split'],
                        vali_split = trial_setup_dict['vali_split'],   
                        currentdir=currentdir,
                        as_dataset = False,
                    )

train_tag = _output_dict['train_tag']
train_features = _output_dict['train_features']
train_values = _output_dict['train_values']

# Build model and optimizer.
model = _optuna.create_model(trial_setup_dict)
optimizer = _optuna.create_optimizer(trial_setup_dict)

ckpt = tf.train.Checkpoint(
    step=tf.Variable(1), optimizer=optimizer, net=model
)
manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)


# Re-use the manager code above ^

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))

import numpy as np

# gen model arch
model_arch = {}
print("Model layers...")
y_pred, _, dense_dict , _INT_dict_pred = model(train_features , training=False, return_LayersInt=True)

for layer_name,layer in dense_dict.items():
    W_b = layer.get_weights()
    if len(W_b) == 2:
        _W,_b = W_b
    elif len(W_b) == 1:
        _W = W_b[0]
        _b = list(np.zeros( _W.shape[1:] ))
    
    if _W.shape == (1,1):
        _W = [ float(_x) for _x in _W[0] ]
    else:
        try:
            _W = [ float(_x) for _x in _W]
            _W = list(np.squeeze(_W))
        except:
            _W = [ [float(_x) for _x in _y] for _y in _W]

    if len(_b) == 1:
        _b = _b[0]
    model_arch[layer_name] = {'weights':_W,'bias': _b}
output_df = None
for _data_type in ['train','vali','test']:
    _tag = _output_dict['{}_tag'.format(_data_type)].values.tolist()
    rct_id = [_x.split("_")[0] for _x in _tag]
    lig_id = [_x.split("_")[1] for _x in _tag]
    _features = _output_dict['{}_features'.format(_data_type)]
    _values = _output_dict['{}_values'.format(_data_type)]

    y_pred, _, dense_dict , _INT_dict_pred = model(_features , training=False, return_LayersInt=True)
    _outputs = {
        "data_type":[_data_type]*len(_tag),
        "tag":_tag,
        "rct_id":rct_id,
        "lig_id":lig_id,
        "y_pred":tf.squeeze(y_pred).numpy(),
        "y_true":_values,
    }
    for k,v in _INT_dict_pred.items():
        _v = tf.squeeze(v).numpy()
        if type(_v[0]) == np.ndarray:
            _v = [ list(_x) for _x in _v ]
        _outputs[k] = _v

        if k in weighting_rule.keys():
            _layer, _idx = weighting_rule[k].split("|")
            _idx = int(_idx)

            weighting = model_arch[_layer]['weights'][_idx]
            _outputs[k+"_Weighted"] = [  weighting*_z for _z in _v]

            

    current_output_df = pd.DataFrame.from_dict(_outputs)

    if output_df is None:
        output_df = current_output_df
    else:
        output_df = output_df.append(current_output_df, ignore_index=True)

_result = {
    "model_arch":model_arch,
}

output_df.sort_values(by=["rct_id","lig_id"])
output_df.to_csv(currentdir+"/"+model_dirname+"/_INT_dict.csv", index=False)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
with open(currentdir+"/"+model_dirname+"/_TrainedModel_INFO","w+") as output:
    output.write(json.dumps(_result,indent=4,cls=NpEncoder))

