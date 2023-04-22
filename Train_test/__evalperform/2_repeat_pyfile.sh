#!/bin/bash

exec_pyfilename='__gen_model_evalVALI.py'
ref_pyfilename='_optuna_VALI.py'
additional_lines='--hide_plt=1'



# get all foldername of models in current dir
model_foldername_list=$(find . -type d -name "_*_model" -maxdepth 1 -exec basename {} \;)
total_num=$(wc -w <<< "$model_foldername_list")
count=1

for model_foldername in $model_foldername_list
do
    echo Progress: $count / $total_num
    count=$[$count+1]
    python ${exec_pyfilename%.*}.py $model_foldername ${ref_pyfilename%.*}.py $additional_lines > _tmp_LOG 
done
