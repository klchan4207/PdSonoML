import os
import json
import re

currentdir = os.getcwd()
checkdir = os.path.join(currentdir,'..')

# check all model folders in dir
subdir_list = [name for name in os.listdir(checkdir) if os.path.isdir(os.path.join(checkdir,name))  if name.endswith('_model')]

# extract all loss info from folders
loss_dict = {}
subdir_list_copy = []
for subdir in subdir_list:
    if '_CONT' in subdir:
        continue
    with open(os.path.join(checkdir,subdir,'loss_record'),'r') as input:
        loss_record = json.loads(input.read())
        loss_dict[subdir] = {
            'loss':loss_record['2']['loss'][-1],
            'vali_loss':loss_record['2']['vali_loss'][-1],
            'test_loss':loss_record['2']['test_loss'][-1],
        }
    subdir_list_copy.append(subdir)

# sort and record the model performance to log file
count = 0
with open(currentdir+'/__DONEmodels_loss_log', 'w') as f:
    for subdir in sorted(subdir_list_copy, key=lambda x: loss_dict[x]['loss']):
        count+=1
        # for tracking 
        print(count, file=f)
        print(subdir, file=f)
        study_num = re.findall(r'_#(\d+)_',subdir)
        if study_num == []:
            study_num = ''
        else:
            study_num = study_num[0]
        print('study',study_num , file=f)
        # loss info
        print(  'loss'   ,  loss_dict[subdir]['loss']   , file=f)
        print('vali_loss',loss_dict[subdir]['vali_loss'], file=f)
        print('test_loss',loss_dict[subdir]['test_loss'], file=f)
