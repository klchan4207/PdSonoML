import tensorflow as tf
from tensorflow.keras import *
from tensorflow.python.keras.engine import input_layer
import json

import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('model_dirname', type=str)
parser.add_argument('_optuna_filename', type=str)
parser.add_argument('--hide_plt', type=int)

currentdir = os.getcwd()
checkdir = os.path.join(currentdir,'..')

import sys
sys.path.insert(1, checkdir+"/../..")
_func = __import__("_func")

model_dirname = parser.parse_args().model_dirname
_optuna_filename = parser.parse_args()._optuna_filename.replace(".py","")
hide_plt = parser.parse_args().hide_plt

with open(checkdir+"/"+model_dirname+"/"+model_dirname+"_setting","r") as input:
    trial_setup_dict = json.load(input)

if "/" in _optuna_filename:
    sys.path.insert(1, os.path.dirname(_optuna_filename) )
    _optuna = __import__(os.path.basename(_optuna_filename))
else:
    _optuna = __import__(_optuna_filename)

# gen data
model_dir = checkdir + "/" + model_dirname + "/" + model_dirname
    
# Get train/test data.
_output_data_dict  = _func.get_data(  
    trial_setup_dict,
    batch_size=trial_setup_dict['batch_size'],
    vali_split=trial_setup_dict['vali_split'],
    test_split=trial_setup_dict['test_split'],
    currentdir=checkdir,
    as_dataset = False
)

train_tag         = _output_data_dict['train_tag']
train_features               = _output_data_dict['train_features']
train_values          = _output_data_dict['train_values']

vali_tag                = _output_data_dict['vali_tag']
vali_features          = _output_data_dict['vali_features']
vali_values                = _output_data_dict['vali_values']

test_tag                = _output_data_dict['test_tag']
test_features          = _output_data_dict['test_features']
test_values                = _output_data_dict['test_values']


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


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
def plot_model(x, y, label_list):
    def hover(event):
        # if the mouse is over the scatter points
        if line.contains(event)[0]:
            # find out the index within the array from the event
            data_pts = line.contains(event)[1]["ind"]
            ind = data_pts[0]
            # get the figure size
            w,h = fig.get_size_inches()*fig.dpi
            ws = (event.x > w/2.)*-1 + (event.x <= w/2.)
            hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0]*ws, xybox[1]*hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy =(x_coord[ind], y_coord[ind])
            # set the image corresponding to that point
            plt.annotate(label_list[ind], (x_coord[ind], y_coord[ind]),color='red',size=15)
            #im.set_data(plt.text(x[ind], y[ind],str(len(data_pts))))
        else:
            #if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    offsetbox = TextArea("-----")

    fig, ax = plt.subplots()
    xybox=(90., 90.)
    ab = AnnotationBbox(offsetbox, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)
    ab.set_visible(False)
    fig.canvas.mpl_connect('motion_notify_event', hover)
    fig = plt.gcf()
    fig.set_size_inches(10.5, 9.5)


    linreg = LinearRegression(normalize=False,fit_intercept=True).fit(x.reshape(-1, 1),y)

    linreg.coef_ = np.array([1])
    linreg.intercept_ = 0
    slope = float(linreg.coef_)
    intercept = float(linreg.intercept_)
    r_square_after = linreg.score(x.reshape(-1, 1), y)
    line_ = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r_square={r_square_after:.6f}'

    print(f"R^2_after = {r_square_after:.6f}")

    x_coord = x
    y_coord = y

    line = plt.scatter(x,y,s=30) #,c=lig_color_list, s=30)
    
    ax.plot(x, intercept + slope * x, label=line_)
    leg = ax.legend(facecolor='white',loc='upper center')

    ax.add_artist(leg)
    if not hide_plt:
        plt.show()

    return r_square_after

_result = {"train":{},"valid":{},"test":{}}
import numpy as np

_features = {"train":train_features,"valid":vali_features,"test":test_features}
_values = {"train":train_values,"valid":vali_values,"test":test_values}
_tag = {"train":train_tag,"valid":vali_tag,"test":test_tag}
for _entry in ["train","valid","test"]:
    # check if the results are consistent or not
    y_pred  = model(_features[_entry] , training=False)
    y_pred = np.array([_x[0] for _x in y_pred])
    y = _values[_entry]
    _tag[_entry] = [str(i) for i in _tag[_entry].values]
    R_square = plot_model(np.array(y_pred), np.array(y), _tag[_entry])
    _result[_entry]['R_square'] = R_square
    RMS = tf.keras.metrics.RootMeanSquaredError()
    MAE = tf.keras.metrics.mean_absolute_error
    MAPE = tf.keras.metrics.mean_absolute_percentage_error
    print("MAE",MAE(y,y_pred).numpy())
    _result[_entry]["MAE"] = float(MAE(y,y_pred).numpy())
    print("MAPE",MAPE(y,y_pred).numpy())
    _result[_entry]["MAPE"] = float(MAPE(y,y_pred).numpy())
    RMS.update_state(y,y_pred)
    print("RMS",RMS.result().numpy() )
    _result[_entry]["RMS"] = float(RMS.result().numpy())
    
    RMS = tf.keras.metrics.RootMeanSquaredError()
    RMS.update_state(np.log10(np.array(y)),np.log10(np.array(y_pred)))
    RMS_result = RMS.result().numpy()
    deviation = 10**RMS_result - 1
    print("deviation",deviation)


with open(checkdir+"/"+model_dirname+"/_eval_result","w+") as output:
    output.write(json.dumps(_result,indent=4))
