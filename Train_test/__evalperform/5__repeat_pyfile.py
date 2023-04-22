import shutil
import os
import pandas as pd

currentdir = os.getcwd()

exec_pyfilename='../__gen_INT_dict.py'
ref_pyfilename='_optuna_VALI.py'
additional_lines=''
listfilename='modellist'

ref_pyfilename = os.path.join(os.path.realpath(os.path.join(currentdir,'..')),ref_pyfilename)

if os.path.splitext(listfilename)[-1] == '.csv':
    _df = pd.read_csv(os.path.join(currentdir,listfilename))
    model_foldername_list = _df['model_foldername'].tolist()
else:
    with open(listfilename,'r') as input:
        model_foldername_list = input.read().split('\n')

_count=1
for model_foldername in model_foldername_list:
    print('           Progress: {} / {}'.format(_count,len(model_foldername_list)))
    _count+=1
    os.system('python {0} {1} {2} {3} > _tmp_LOG '.format(exec_pyfilename,model_foldername,ref_pyfilename,additional_lines))
    shutil.copytree(currentdir+'/../'+model_foldername,currentdir+'/'+model_foldername, dirs_exist_ok=True)
os.system('rm _tmp_LOG')