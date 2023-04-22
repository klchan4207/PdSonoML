#!/bin/bash

exec_pyfilename='__gen_int_dict.py'
ref_pyfilename='_optuna_VALI.py'
additional_lines=''
listfilename='modellist'



# get all foldername of models in current dir
model_foldername_list=$(cat  $listfilename |tr "\n" " ")
total_num=$(wc -w <<< "$model_foldername_list")
count=1

for model_foldername in $model_foldername_list
do
    echo Progress: $count / $total_num
    count=$[$count+1]
    python ${exec_pyfilename%.*}.py $model_foldername ${ref_pyfilename%.*}.py $additional_lines > _tmp_LOG 
done
