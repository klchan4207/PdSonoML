import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import mpld3
import numpy as np

import os
currentdir = os.getcwd()
#currentdir = os.path.dirname(os.path.realpath(__file__))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modellist', type=str)
parser.add_argument('--include_CONT', type=int)
parser.add_argument('--mode', type=str)

#outliner_list = [15,28]

_modellist = parser.parse_args().modellist
_include_CONT = parser.parse_args().include_CONT
_mode = parser.parse_args().mode

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
    "lig":{},
    "rctXlig":{}
}

_sample = model_folder_list[0]
_sample_df = pd.read_csv(currentdir+"/"+_sample+'/_INT_dict.csv')

lig_id_list = sorted(list(set(_sample_df['lig_id'].values.tolist())))
rct_id_list = sorted(list(set(_sample_df['rct_id'].values.tolist())))



# source: https://onestopdataanalysis.com/python-outlier-detection/
def detect_outlier(data):
    # find q1 and q3 values
    q1, q3 = np.percentile(sorted(data), [25, 75])

    # compute IRQ
    iqr = q3 - q1

    # find lower and upper bounds
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = [(idx,x) for idx, x in enumerate(data) if x <= lower_bound or x >= upper_bound]
 
    return [_x[0] for _x in outliers], [_x[1] for _x in outliers]


for lig_id in lig_id_list:
    _output_dict["lig"][lig_id] = []
    _output_dict["rctXlig"][lig_id] = {}
    for rct_id in rct_id_list:
        _output_dict["rctXlig"][lig_id][rct_id] = []
count = 1
for model_folder in model_folder_list:
    _INT_df = pd.read_csv(currentdir+"/"+model_folder+'/_INT_dict.csv')
    for lig_id in lig_id_list:
        lig_E_value = _INT_df.loc[_INT_df['lig_id']==lig_id]['inputs_E2_Weighted'].values[0]
        _output_dict["lig"][lig_id].append(lig_E_value)
        for rct_id in rct_id_list:
            rctXlig_E_value = _INT_df.loc[(_INT_df['lig_id']==lig_id) & (_INT_df['rct_id']==rct_id)]['inputs_E1_Weighted'].values[0]
            _output_dict["rctXlig"][lig_id][rct_id].append(rctXlig_E_value)


saving_dir = currentdir+"/batchcheck_INT"

if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

    
from statistics import mean
import numpy as np
ligE_avg_list = []
for k,v in _output_dict["lig"].items():
    ligE_avg_list.append( mean(_output_dict["lig"][k]))
lig_id_list_sorted = np.argsort(ligE_avg_list)+1

plot_set = [
    (lig_id_list,""),
    (lig_id_list_sorted,"_sorted"),
]
_plot_dict = {}
for plot_id_list,_suffix in plot_set:

    x_labels = [str(i) for i in plot_id_list]
    data_set = [ _output_dict["lig"][lig_id] for lig_id in  plot_id_list]
    fig = plt.figure(figsize =(10, 7))
    top = int(max([i for i in max([_x for _x in data_set]) ]))+1
    bottom = 0 #int(min([i for i in min([_x for _x in data_set]) ]))-1

    if _mode == None:
        plt.plot(data_set,'x')
        plt.legend([i+1 for i in range(len(model_folder_list))],loc="upper right")
        plt.xticks([i for i in range(len(x_labels))], x_labels, fontsize=20)
    elif _mode == "boxplot":
        plt.boxplot(data_set)
        plt.xticks([i+1 for i in range(len(x_labels))], x_labels, fontsize=20)

    from matplotlib.ticker import StrMethodFormatter
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    plt.rc('ytick', labelsize=20)
    title = "ligE{}".format(_suffix)
    #plt.title(title)
    plt.ylim(top=top)
    plt.ylim(bottom=bottom)

    plt.xlabel("Ligands", fontsize=26)
    #plt.ylabel("ΔG‡(L) (kcal mol⁻¹)", fontsize=18)

    html_str = mpld3.fig_to_html(fig)
    Html_file= open(saving_dir+"/_INT_lig{}.html".format(_suffix),"w")
    Html_file.write(html_str)
    Html_file.close()

    _plot_dict[title] = {
        "dataset":data_set,
        "order":plot_id_list,
    }
    plt.savefig(saving_dir+"/_INT_lig{}.png".format(_suffix))
    ##plt.show()



rctXligE_avg_list = {}
for lig_id,_dict in _output_dict["rctXlig"].items():
    rctXligE_avg_list[lig_id] = {}
    for rct_id,v in _dict.items():
        rctXligE_avg_list[lig_id][rct_id] = mean(_output_dict["rctXlig"][lig_id][rct_id])




Hammett_constant = [0.78,0.7,0.52,0.49,0.47,0.36,0.45,0.37,0.46,0.53,0.34,0.06,0,-0.06,-0.14,0.11,-0.28,-0.15,-0.15,-0.83]
y_unsorted = [ rctXligE_avg_list[7][rct_id] for rct_id in  rct_id_list]
plot_id_list = [_x for _,_x in sorted(zip(y_unsorted,rct_id_list))]
data_set = [ _output_dict["rctXlig"][7][rct_id] for rct_id in  plot_id_list]
fig = plt.figure(figsize =(10, 7))
top = (int(max([i for i in max([_x for _x in data_set]) ])*2)+1)/2
bottom = int(min([i for i in min([_x for _x in data_set]) ])*2)/2
if _mode == None:
    plt.plot(data_set,'x')
    plt.legend([i+1 for i in range(len(model_folder_list))],loc="upper right")
    plt.yticks(np.arange(bottom,top+0.5, 0.5))
    plt.xticks([i for i in range(len(plot_id_list))], plot_id_list, fontsize=20)
elif _mode == "boxplot":
    plt.boxplot(data_set)
    plt.yticks(np.arange(bottom,top+0.5, 0.5))
    plt.xticks([i+1 for i in range(len(plot_id_list))], plot_id_list, fontsize=20)
plt.ylim(top=top)
plt.ylim(bottom=bottom)
plt.rc('ytick', labelsize=20)
title = "rctXlig of ligand 7"
plt.title(title)
plt.xlabel("Reactants", fontsize=26)
html_str = mpld3.fig_to_html(fig)
Html_file= open(saving_dir+"/_INT_rctXligE_lig7.html","w")
Html_file.write(html_str)
Html_file.close()
plt.savefig(saving_dir+"/_INT_rctXligE_lig7.png")


prev_top = top
prev_bottom = bottom
Hammett_constant = [0.78,0.7,0.52,0.49,0.47,0.36,0.45,0.37,0.46,0.53,0.34,0.06,0,-0.06,-0.14,0.11,-0.28,-0.15,-0.15,-0.83]
current_rct_id = 1

y_unsorted = [ rctXligE_avg_list[lig_id][current_rct_id] for lig_id in  lig_id_list]
plot_id_list = [_x for _,_x in sorted(zip(y_unsorted,lig_id_list))]


data_set = [ _output_dict["rctXlig"][lig_id][current_rct_id] for lig_id in  plot_id_list]

fig = plt.figure(figsize =(10, 7))
top = (int(max([i for i in max([_x for _x in data_set]) ])*2)+1)/2
bottom = int(min([i for i in min([_x for _x in data_set]) ])*2)/2
top = prev_top
bottom = prev_bottom
if _mode == None:
    plt.plot(data_set,'x')
    plt.legend([i+1 for i in range(len(model_folder_list))],loc="upper right")
    plt.yticks(np.arange(bottom,top+0.5, 0.5))
    plt.xticks([i for i in range(len(plot_id_list))], plot_id_list, fontsize=20)
elif _mode == "boxplot":
    plt.boxplot(data_set)
    plt.yticks(np.arange(bottom,top+0.5, 0.5))
    plt.xticks([i+1 for i in range(len(plot_id_list))], plot_id_list, fontsize=20)
plt.rc('ytick', labelsize=20)
title = "rctXligE_rct1"
plt.ylim(top=top)
plt.ylim(bottom=bottom)
plt.title(title)
#plt.ylim(top=top)
#plt.ylim(bottom=bottom)

plt.xlabel("Ligands", fontsize=26)

html_str = mpld3.fig_to_html(fig)
Html_file= open(saving_dir+"/_INT_rctXligE_rct1.html","w")
Html_file.write(html_str)
Html_file.close()

plt.savefig(saving_dir+"/_INT_rctXligE_rct1.png")

import pickle
with open(saving_dir+'/_INT_lig.pickle', 'wb') as handle:
    pickle.dump(_plot_dict, handle)
