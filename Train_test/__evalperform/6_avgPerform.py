import tensorflow as tf
from tensorflow.keras import *
import json
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('result_filename', type=str)
parser.add_argument('--hide_plt', type=int)
parser.add_argument('--round_digits', type=int, default=2)
result_filename = parser.parse_args().result_filename.replace('.csv','')+'.csv'
hide_plt = parser.parse_args().hide_plt
round_digits = parser.parse_args().round_digits

import os
currentdir = os.getcwd()
#currentdir = os.path.dirname(os.path.realpath(__file__))

def plot_model(x, y, _entry):
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings('ignore') 

    plt.rc('xtick', labelsize=32)
    plt.rc('ytick', labelsize=32)
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)

    # for R^2
    linreg = LinearRegression(normalize=False,fit_intercept=True).fit(x.reshape(-1, 1),y)
    linreg.coef_ = np.array([1])
    linreg.intercept_ = 0
    R_square = linreg.score(x.reshape(-1, 1), y)

    # plot points
    plt.scatter(x,y,s=10)

    # plot line
    _plot_min = 0
    _plot_max = 50
    ax.plot([_plot_min-1, _plot_max+1], [_plot_min-1, _plot_max+1], 'k--', lw=1.75)
    plt.xlim(_plot_min,_plot_max)
    plt.ylim(_plot_min,_plot_max)

    # label for x & y axis
    plt.xlabel('Experimental rate constant\n k (1000/h⁻¹)', fontsize=36)
    plt.ylabel('Predicted rate constant\n k (1000/h⁻¹)', fontsize=36)

    # legend
    if R_square<0:
        _legend =    'R²    < 0'
    else:
        _legend =    'R²    = {0}'.format(round(R_square,round_digits))
    _legend += '\nMAE = {0:.{1}f}'.format(round(_result[_entry]['MAE'],round_digits),round_digits)
    _legend += '\nRMS = {0:.{1}f}'.format(round(_result[_entry]['RMS'],round_digits),round_digits)
    from matplotlib.offsetbox import AnchoredText
    at = AnchoredText(
        _legend, prop=dict(size=36), frameon=True, loc='lower right')
    ax.add_artist(at)

    # index and titles
    fig.text(-0.05, 0.95, 
        {   
            'train': 'a)',
            'vali': 'b)',
            'test': 'c)'
        }[_entry], 
        horizontalalignment='left',
        verticalalignment='top',
        size=45)
    plt.title(
        {
            'train': 'Training Set Performance',
            'vali': 'Validation Set Performance',
            'test': 'Testing Set Performance'
        }[_entry],
        size=35,
        pad=25)    

    # save figure
    plt.savefig(saving_dir+'/{}.png'.format(_entry), bbox_inches='tight',dpi=500)

    if not hide_plt:
        plt.show()

    return R_square

def plotall_model(xdict, ydict):
    import matplotlib.pyplot as plt    

    plt.rc('xtick', labelsize=32)
    plt.rc('ytick', labelsize=32)
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.set_size_inches(10, 10)

    _scattersize = 15
    for k, x in xdict.items():
        y = ydict[k]
        if k == "train":
            _train_scat = plt.scatter(x,y,s=_scattersize,c='red', marker='o')
        elif k == "vali":
            _vali_scat = plt.scatter(x,y,s=_scattersize,c='green', marker='x')
        elif k == "test":
            _test_scat = plt.scatter(x,y,s=_scattersize,c='blue', marker='^')
    lgnd = plt.legend((_train_scat, _vali_scat, _test_scat),
           ('Training set', 'Validation set', 'Testing set'),
           scatterpoints=1,
           loc='lower right',
           #ncol=3,
           fontsize=20)
    for _i in range(3):
        lgnd.legendHandles[_i]._sizes = [50]

    _plot_min = 0
    _plot_max = 50

    ax.plot([_plot_min-1, _plot_max+1], [_plot_min-1, _plot_max+1], 'k--', lw=1.75)

    plt.xlabel("Experimental rate constant\n k (1000/h⁻¹)", fontsize=36)
    plt.ylabel("Predicted rate constant\n k (1000/h⁻¹)", fontsize=36)

    plt.xlim(_plot_min,_plot_max)
    plt.ylim(_plot_min,_plot_max)

    title = 'Performance of \nTraining/Validation/Testing Sets'

    plt.title(title,size=35,pad=25)    

    plt.savefig(saving_dir+"/ALL.png", bbox_inches="tight",dpi=450)
    
    if not hide_plt:
        plt.show()

saving_dir = currentdir+'/avgPerform'
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

input_df = pd.read_csv(currentdir+'/'+result_filename)
_result = {}
for _x in ['train','vali','test']:
    _result[_x]={
        'values':input_df.loc[input_df['data_type']==_x]['y_true'].values.tolist(),
        'pred':input_df.loc[input_df['data_type']==_x]['y_pred'].values.tolist(),
    }

_ALL_xdict = {} ; _ALL_ydict = {}
for _entry in ['train','vali','test']:

    y = np.array(_result[_entry]['values'])
    y_pred  = np.array(_result[_entry]['pred'])


    MAE = tf.keras.metrics.mean_absolute_error
    _result[_entry]['MAE'] = float(MAE(y,y_pred).numpy())

    RMS = tf.keras.metrics.RootMeanSquaredError()
    RMS.update_state(y,y_pred)
    _result[_entry]['RMS'] = float(RMS.result().numpy())

    R_square = plot_model(y, y_pred, _entry)
    _result[_entry]['R_square'] = R_square

    _ALL_xdict[_entry] = y
    _ALL_ydict[_entry] = y_pred

plotall_model(_ALL_xdict, _ALL_ydict)

# save result
with open(saving_dir+'/avgPerform_LOG','w+') as output:
    output.write(json.dumps(_result,indent=4))

