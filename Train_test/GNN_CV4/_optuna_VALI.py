import optuna
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import dgl

import pandas as pd
import numpy as np
import os
import time
import pickle
import datetime , time
import json
import copy
import math
import argparse
import glob
import random
import sys

from sklearn.linear_model import LinearRegression

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, currentdir+"/../../Model")
sys.path.insert(1, currentdir+"/../..")
_func = __import__("_func")

#________________________________________________________________________________________________________

# get cpu/gpu
tf.debugging.set_log_device_placement(True)
_detect_gpus = tf.config.list_logical_devices('GPU')
if len(_detect_gpus)==0:
    _device_type="cpu"
else:
    _device_type="gpu"


#________________________________________________________________________________________________________

# Helper function to get data
get_data =  _func.get_data

# model defining
def create_model(trial_setup_dict): # normalizer always = None
    print("Model ...")
        
    GNN_model = __import__(trial_setup_dict['modeltype_pyfilename'])

    model = GNN_model.MODEL(trial_setup_dict['model'])

    return model

def create_optimizer(trial_setup_dict):
    print("Optimizer ...")
    optimizer_chosen = trial_setup_dict["optimizer_chosen"]
    kwargs = trial_setup_dict[optimizer_chosen]

    optimizer = getattr(tf.optimizers, optimizer_chosen)(**kwargs)
    return optimizer

def learn(model, 
            optimizer, 
            train_dataset, 
            loss_mode,
            mode="eval", 
            feature_type="normal" , 
            training_seed = None, 
            _device_id="0"):
    MAE = tf.keras.losses.MeanAbsoluteError()
    MAPE = tf.keras.losses.MeanAbsolutePercentageError()

    epoch_loss = 0
    total = 0

    if mode == "train" and training_seed == None:
        print("ERROR, require to set a training seed")
        exit()

    # turn off
    # expect the input to be sth like _1
    #_device_id=_device_id.replace("_","")
    #if _device_id=="":
    #    _device_id = "0"
    _device_id = "0"
    _device = "/{0}:{1}".format(_device_type,_device_id)
    with tf.device(_device):
        for batch, (features, labels) in enumerate(train_dataset):
            labels = tf.convert_to_tensor(labels)
            with tf.GradientTape() as tape:
                # set seed for reproducibility
                if mode == "train":
                    tf.random.set_seed(training_seed)
                logits = model(features, training=(mode=="train"), _device=_device)
                if loss_mode == 'MAE':
                    loss = MAE(logits,labels)
                elif loss_mode == 'MAPE':
                    loss = MAPE(labels,logits)
                
            if feature_type == "normal":
                current_batch_size = len(features)
            else:
                current_batch_size = len(features[0])

            total += current_batch_size
            epoch_loss = (epoch_loss*(total-current_batch_size) + loss*current_batch_size)/total
            gradients = tape.gradient(loss, model.trainable_variables)
            if (mode=="train"):
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    if mode == "eval":
        print("Eval ...")
    elif mode == "eval_silent":
        pass
    return epoch_loss

def evaluate(model, optimizer, test_dataset,trial_setup_dict):

    
    method = trial_setup_dict["eval_method"]

    if method == "default":
        loss_mode=trial_setup_dict['loss_mode']
        return learn(model, optimizer, test_dataset, loss_mode, "eval_silent", feature_type = 'split' )

    elif method == "r_square":
        # pretreatment
        test_features = np.concatenate([x for _, (x, _) in enumerate(test_dataset)])
        test_values = np.concatenate([y for _, (_, y) in enumerate(test_dataset)])
        pred = model.predict(np.array(test_features))
        real = test_values
        _pred = np.array(pred).reshape(-1, 1)
        _real = np.array(real)
        linreg = LinearRegression(normalize=False,fit_intercept=True).fit(_pred,_real)
        linreg.coef_ = np.array([1])
        linreg.intercept_ = 0 
        get_r_square = linreg.score(_pred, _real)
        return get_r_square




def objective(trial,optuna_setup_dict, mode="trial"):
    print("Objective ...")

    if mode=="trial":
        #initialize trial
        for k,v in optuna_setup_dict.items():
            # treat numbers with range
            if type(v)==dict:
                v_type = list(v.keys())[0]
                if v_type == 'categorical':
                    v_choices = v[v_type]
                    #if type(v_choices) == list:
                    trial.suggest_categorical(k,v_choices )
                    #else:
                    #    print("##### syntax ERROR #####, at:",k,v)
                elif v_type == 'discrete_uniform':
                    v_low = v[v_type][0]
                    v_high = v[v_type][1]
                    v_q = v[v_type][2]  
                    trial.suggest_discrete_uniform(k, v_low,v_high,v_q)
                elif v_type in ['float','int']:
                    v_low = v[v_type][0]
                    v_high = v[v_type][1]
                    if v_type == 'float': v_step = None
                    elif v_type == 'int': v_step = 1
                    v_log = False
                    if len(v[v_type]) > 2:
                        v_optionals = v[v_type][2]
                        v_step = v_optionals.get('step')
                        v_log = v_optionals.get('log')
                        if v_log == None:
                            v_log = False
                    if v_step and v_log:
                        print("ERROR at {}, step and log cannot be used together:".foramt(v_type),k,v)
                        exit()
                    else:
                        if v_type == 'float':
                            trial.suggest_float(k, v_low, v_high, step=v_step,log=v_log)
                        elif v_type == 'int':
                            trial.suggest_int(k, v_low, v_high, step=v_step,log=v_log)
                elif v_type in ['loguniform','uniform']:
                    v_low = v[v_type][0]
                    v_high = v[v_type][1]
                    if v_type == 'loguniform':
                        trial.suggest_loguniform(k, v_low, v_high)
                    elif v_type == 'uniform':
                        trial.suggest_uniform(k, v_low, v_high)
                else:
                    print("ERROR v_type {}",k,v)
                print("setting {} to param {} ".format(v,k))
            else:
                print("setting {} to attr {} ".format(v,k))
                trial.set_user_attr(k, v)


        def merge(d1, d2):
            for k in d2:
                if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict):
                    merge(d1[k], d2[k])
                else:
                    d1[k] = d2[k]
        def copy_merge(d1,d2):
            _d1 = copy.deepcopy(d1)
            _d2 = copy.deepcopy(d2)
            merge(_d1,_d2)
            return _d1

        trial_setup_dict = {}
        for _dict in [trial.user_attrs , trial.params]:
            for k,v in _dict.items():
                if "|" in k:
                    _kname_list = k.split("|")
                    _kname_list.reverse()
                    _dict = v
                    for _kname in _kname_list:
                        _dict = {_kname:_dict }
                    trial_setup_dict = copy_merge(trial_setup_dict,_dict)
                else:
                    trial_setup_dict[k] = v
    elif mode=="simple_train" or  mode=="cont_train":
        trial_setup_dict =  copy.deepcopy(optuna_setup_dict)


    # choose optuna mode
    opttrain_mode = trial_setup_dict['opttrain_mode']
    #opttrain = __import__(opttrain_mode)

    # refine trial inputs
    trial_setup_dict['input_ndata_dim'] = len(trial_setup_dict['input_ndata_list'])
    trial_setup_dict['input_edata_dim'] = len(trial_setup_dict['input_edata_list'])
    devid_suffix = trial_setup_dict['devid_suffix']

    #________________________________________________________________________________________________________
    # define saving directory
    z = datetime.datetime.now()
    study_foldername = trial_setup_dict['study_foldername']
    if mode != "cont_train": #trial_setup_dict.get('modeltrained_foldername') == None:
        modeltrained_foldername = "_{0}_{1}_{2:02d}_{3:02d}_s{4:02d}_ms{5:06d}_model".format(study_foldername,z.date(),z.hour,z.minute,z.second,z.microsecond).replace("-","")
        trial_setup_dict['modeltrained_foldername'] = modeltrained_foldername
    else:
        loss_mode = trial_setup_dict['loss_mode']
        modeltrained_foldername_prev = trial_setup_dict['modeltrained_foldername']
        trial_setup_dict['modeltrained_foldername'] = modeltrained_foldername_prev+"_##{}_CONT".format(loss_mode)
        modeltrained_foldername = trial_setup_dict['modeltrained_foldername']
    saving_dir = currentdir+"/"+modeltrained_foldername


    stdoutOrigin=sys.stdout 
    if trial_setup_dict.get('_test_mode') == "1":
        log_filename = "_TEST_trainlog"
    else:
        log_filename = modeltrained_foldername+"_trainlog"

    sys.stdout = open(log_filename, "w")

    #________________________________________________________________________________________________________
    # gen seed if not specified
    if trial_setup_dict.get('training_seed') == None:
        trial_setup_dict['training_seed'] = random.randint(0,9999)
    training_seed = trial_setup_dict['training_seed']

    print('trial_setup_dict')
    print(json.dumps(trial_setup_dict,indent=4))

    print('training_seed=',training_seed)


    #________________________________________________________________________________________________________
    trial_begin = time.time()
    # Get train/test data.
    _output_data_dict = _func.get_data(
        trial_setup_dict,
        batch_size=trial_setup_dict['batch_size'],
        test_split=trial_setup_dict['test_split'],
        vali_split=trial_setup_dict['vali_split'],
        currentdir = currentdir,
        return_indexes=True
    )
    dummy_train_dataset         = _output_data_dict['stage1_train_dataset']
    train_dataset               = _output_data_dict['train_dataset']
    dummy_vali_dataset          = _output_data_dict['stage1_vali_dataset']
    vali_dataset                = _output_data_dict['vali_dataset']
    dummy_test_dataset          = _output_data_dict['stage1_test_dataset']
    test_dataset                = _output_data_dict['test_dataset']

    node_num_info               = _output_data_dict['node_num_info']
    train_vali_test_indexes     = _output_data_dict['train_vali_test_indexes']

    if trial_setup_dict.get('node_num_info') == None:
        trial_setup_dict['node_num_info'] = node_num_info

    # Build model and optimizer.
    model = create_model(trial_setup_dict)
    optimizer = create_optimizer(trial_setup_dict)

    # setup for saving models
    if mode=="cont_train":
        print("...Restoring checkpoint")
        # restore ckpt
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=optimizer, net=model
        )
        saving_dir_prev = currentdir+"/{}/{}".format(modeltrained_foldername_prev,modeltrained_foldername_prev)
        _manager = tf.train.CheckpointManager(ckpt, saving_dir_prev, max_to_keep=3)
        ckpt.restore(_manager.latest_checkpoint)
        if _manager.latest_checkpoint:
            print("Restored from {}".format(_manager.latest_checkpoint))
    else:
        # create ckpt
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=optimizer, net=model
        )
    manager = tf.train.CheckpointManager(ckpt, saving_dir+"/{}".format(saving_dir.split("/")[-1]
    ) 
        , max_to_keep=3)

    # gen cutoff
    stage1_cutoff = trial_setup_dict['stage1_cutoff']
    stage1_skip = trial_setup_dict.get('stage1_skip')
    stage2_converge = trial_setup_dict.get('stage2_converge')

    stage2_fluctuate_dv = trial_setup_dict['stage2_fluctuate_dv']
    stage2_fluctuate_pc = trial_setup_dict['stage2_fluctuate_pc']
    if stage2_converge == None:
        stage2_converge = 0.0
    loss_mode = trial_setup_dict['loss_mode']

    loss_record_dict = {1:{},2:{}}
    # itterate training sequence:
    for _stage in [1,2]:
        max_epoch_num = trial_setup_dict['max_epoch_num']

        if _stage==1:
            print("################## STAGE  1 #######################")
            model.stopuseArrheniusEq()
            trial_train_dataset = dummy_train_dataset
            trial_vali_dataset = dummy_vali_dataset
            trial_test_dataset = dummy_test_dataset
            if stage1_skip:
                print("Skipping stage 1")
                continue
        if _stage==2:
            print("################## STAGE  2 #######################")
            model.startuseArrheniusEq()
            trial_train_dataset = train_dataset
            trial_vali_dataset = vali_dataset
            trial_test_dataset = test_dataset
        print("use Arrhenius:",model.useArrheniusEq)


        print("Epoch ( [train|valid|test] ): ", end =" ", flush=True)
        # Training and testing cycle.
        last10_loss = np.array([i*10 for i in range(1,11)]) # [10,... 100]
        loss_record = {"loss":[],"vali_loss":[],"test_loss":[]}

        for i in range(max_epoch_num):
            loss = learn(model, optimizer, trial_train_dataset, loss_mode, "train", feature_type = 'split' , training_seed = training_seed) #, _device_id=devid_suffix)
            loss_record['loss'].append(round(float(loss),7))
            
            vali_loss = learn(model, optimizer, trial_vali_dataset, loss_mode, "eval_silent", feature_type = 'split'
            )
            loss_record['vali_loss'].append(round(float(vali_loss),7))

            test_loss = learn(model, optimizer, trial_test_dataset, loss_mode, "eval_silent", feature_type = 'split'
            )
            loss_record['test_loss'].append(round(float(test_loss),7))

    
            ckpt.step.assign_add(1)

            if _stage==1 and round(float(loss),7) <=stage1_cutoff:
                break

            if _stage==2:
                last10_loss_prev = last10_loss.copy()
                last10_loss = last10_loss[1:]
                last10_loss = np.append(last10_loss,[loss])

                delta_loss = [last10_loss[_i]-last10_loss_prev[_i] for _i in range(len(last10_loss_prev))]
                fluctuate_pc = sum([  0 > delta_loss[_i]*delta_loss[_i+1] for _i in range(len(delta_loss)-1)]) / (len(delta_loss)-1)
                avg_dv = np.mean(np.abs(delta_loss))
                
                # break if converged
                if avg_dv <= stage2_converge:
                    print("...converged")
                    break
                # break if fluctuate too much
                if fluctuate_pc >= stage2_fluctuate_pc and avg_dv >= stage2_fluctuate_dv:
                    print("...fluctuation too high:")
                    print("fluctuate_pc:",fluctuate_pc)
                    print("stage2_fluctuate_dv:",avg_dv)
                    break

            if i%10 == 0:
                print("{} [{}|{}|{}]-> ".format(i,round(float(loss),7) , round(float(vali_loss),7), round(float(test_loss),7) ), end =" ", flush=True)

        print("{} [{}|{}|{}]-> ".format(i,round(float(loss),7) , round(float(vali_loss),7), round(float(test_loss),7) ), end =" ", flush=True)
        print("END")
        if _stage==2:
            print("avg_dv out of {0}: {1}".format(len(last10_loss),avg_dv))
        loss_record_dict[_stage] = loss_record

    eval_result = evaluate(model, optimizer, test_dataset, trial_setup_dict)
    trial_end = time.time()



    # save model
    manager.save()
    with open(saving_dir+"/{}_setting".format(saving_dir.split("/")[-1]
    ),"w+") as f:
        f.write(json.dumps(trial_setup_dict,indent=4).replace("\'","\""))
    with open(saving_dir+"/{}_train_vali_test_indexes".format(saving_dir.split("/")[-1]
    ),"w+") as f:
        f.write(json.dumps(train_vali_test_indexes,indent=4).replace("\'","\""))
    with open(saving_dir+"/loss_record","w+") as f:
        f.write(json.dumps(loss_record_dict,indent=4).replace("\'","\""))


    # save other files
    with open(__file__,"r") as input:
        raw_pyfile_record = input.read()

    with open(saving_dir+"/rawpyrecord","w+") as f:
        f.write(raw_pyfile_record)
        
    print("This trial used {} seconds".format(trial_end-trial_begin))
    
    # save logfiles
    sys.stdout.close()
    sys.stdout=stdoutOrigin
    with open(log_filename, "r") as input:
        _log = input.read()
    with open(saving_dir+"/trainlog", "w+") as output:
        output.write(_log)
    os.remove(log_filename)

    return eval_result


def search(optuna_setup_dict,study):

    print(f"Sampler used is {study.sampler.__class__.__name__}")

    f = lambda y: objective(y,optuna_setup_dict)

    study_filename = optuna_setup_dict["study_foldername"]
    study_dir = os.path.join(currentdir,study_filename,"study")

    total_n_trials = optuna_setup_dict["study_total_n_trials"]-len(study.trials)
    num_per_batch =  optuna_setup_dict["study_num_per_batch"]
    total_batch_count = math.ceil(total_n_trials/num_per_batch)

    record_study_log = optuna_setup_dict["study_log_input"]
    if record_study_log:
        study_log_inputdir = os.path.join(currentdir,optuna_setup_dict["study_log_input"])
        study_log_outputdir = os.path.join(currentdir,optuna_setup_dict["study_log_output"])
    
    for _i in range(1,total_batch_count+1):

        n_trials = 3
        if _i*num_per_batch > total_n_trials :
            n_trials = total_n_trials%num_per_batch

        if os.path.isfile(study_dir):
            with open(study_dir, 'rb') as _study_input:
                study = pickle.load(_study_input)

        study.optimize(f, n_trials=n_trials)

        # save study
        pickle.dump(study, open(study_dir, "wb"))

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value)) 

        if record_study_log:
            with open(study_log_inputdir,"r") as _study_log_input:
                study_log = _study_log_input.read()
            with open(study_log_outputdir,"w+") as _study_log_output:
                _study_log_output.write(study_log)
        




if __name__ == '__main__':  

    parser = argparse.ArgumentParser()
    parser.add_argument('new_filenames', type=str, nargs='+')
    parser.add_argument('--devid', type=int)
    parser.add_argument('--cont_study', type=str)
    parser.add_argument('--study_log', type=str)
    parser.add_argument('--test_mode', type=str)

    _new_filenames = parser.parse_args().new_filenames
    _devid = parser.parse_args().devid
    _cont_study = parser.parse_args().cont_study
    _study_log = parser.parse_args().study_log
    _test_mode =  parser.parse_args().test_mode



    # get current filelog
    if _devid:
        devid_suffix = "_#{}".format(_devid)
    else:
        devid_suffix = ""

    if _cont_study:
        print("...continuing Study")
        # get continue study file & setup filename
        study_foldername = _cont_study
        study_dir = os.path.join(currentdir,study_foldername)
        if _test_mode != "1":
            with open(os.path.join(study_dir,"_setup.inp"), "r") as input:
                optuna_setup_dict = json.loads(str(input.read()))

        # save folder name
        print(optuna_setup_dict)
        optuna_setup_dict["study_foldername"] = study_foldername
        # load study
        with open(os.path.join(study_dir,"study"), 'rb') as _study_input:
            study = pickle.load(_study_input)

    elif _new_filenames:
        print("...creating new Study")
        # get setup filename
        setup_filename = _new_filenames[0].replace(".inp","")
        with open(setup_filename+".inp", "r") as input:
            optuna_setup_dict = json.loads(str(input.read()))
    
        # create study folder
        all_begin = time.time()
        z = datetime.datetime.now()
        study_foldername = "Optuna_StudyResult_{0}_{1:02d}{2:02d}{3}".format(z.date(),z.hour,z.minute,devid_suffix).replace("-","")
        optuna_setup_dict["study_foldername"] = study_foldername
        study_dir = currentdir+"/"+study_foldername
        if _test_mode != "1":
            if not( os.path.exists(study_dir) ):
                os.makedirs(study_dir)
            with open(os.path.join(currentdir,study_foldername,"_setup.inp"), "w+") as output:
                output.write(json.dumps(optuna_setup_dict,indent=4))

        # gen seed for sampler
        if optuna_setup_dict.get('sampler_seed') == None:
            sampler_seed = random.randint(0,9999)
            optuna_setup_dict['sampler_seed'] = sampler_seed
        sampler_seed = optuna_setup_dict['sampler_seed']

        # create study
        direction = optuna_setup_dict['study_direction']
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction=direction,
                                    pruner=optuna.pruners.HyperbandPruner(),
                                    sampler=sampler
                                    )
    # save others
    optuna_setup_dict['devid_suffix'] = devid_suffix
    optuna_setup_dict['study_log_input'] = _study_log
    optuna_setup_dict['study_log_output']  = 'study_log{}'.format(len(glob.glob(study_dir+"/study_log*"))+1)
    optuna_setup_dict['_test_mode'] = str(_test_mode)

    with open(__file__,"r") as input:
        print(input.read())
    search(optuna_setup_dict,study)
