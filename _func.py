import os
import pandas as pd
import numpy as np
import tensorflow as tf
import dgl
import copy


def get_data(   trial_setup_dict,
                batch_size,
                vali_split, 
                test_split, 
                as_dataset = True,
                return_indexes=False,
                currentdir=None,
                #normalize = True, 
                #masking_featurename=None,
                unique_num_dict = None,
            ):
    print("IMPORTING from _func ...")
    print("Data ...")

    # for selecting certain list as training set
    # ---------------------------------------------------------------------------------------------
    vali_tag = trial_setup_dict.get("vali_selected_tag")
    test_tag = trial_setup_dict.get("test_selected_tag")
    # for selecting certain rct or lig as testing set or training set
    # ---------------------------------------------------------------------------------------------
    _id_dict={
            "train":{
                "rct":[],
                "lig":[],
            },
            "vali":{
                "rct":[],
                "lig":[],
            },
            "test":{
                "rct":[],
                "lig":[],
            }
        }
    _EMPTY_id_dict= copy.deepcopy(_id_dict)
    for _x in ["test","vali","train"]:
        for _y in ["rct","lig"]:
            _value = trial_setup_dict.get("{0}_{1}_id_list".format(_x,_y))
            if _value != None:
                _id_dict[_x][_y] = _value
    # ---------------------------------------------------------------------------------------------

    if currentdir == None:
        datadir = os.path.dirname(os.path.realpath(__file__))+'/../../DATA/'
    else:
        datadir = currentdir+'/../../DATA/'

    # get data from files---------------------------------------------------------------------------------------------
    
    # Ligand data______________________________________________________________
    # edge, node datafiles
    LIG_EDGEdataset_filename = trial_setup_dict.get("dataset_edge(LIG)_filename")
    if LIG_EDGEdataset_filename:
        LIG_EDGEdataset =   pd.read_csv(datadir + LIG_EDGEdataset_filename)
    else:
        LIG_EDGEdataset =   None

    LIG_NODEdataset_filename = trial_setup_dict.get("dataset_node(LIG)_filename")
    if LIG_NODEdataset_filename:
        LIG_NODEdataset =   pd.read_csv(datadir + LIG_NODEdataset_filename)
    else:
        LIG_NODEdataset =   None

    # Extra datafiles
    LIG_EXTRAdataset_filename = trial_setup_dict.get("dataset_extra(LIG)_filename")
    if LIG_EXTRAdataset_filename:
        LIG_EXTRAdataset =  pd.read_excel(datadir + LIG_EXTRAdataset_filename)
    else:
        LIG_EXTRAdataset =  None
    
    # specify the mode
    LIG_data_mode = trial_setup_dict.get("LIG_data_mode")
    if LIG_data_mode == None:
        LIG_data_mode = "graphONLY"

    # Reactant data______________________________________________________________
    RCT_dataset_filename = trial_setup_dict["dataset(RCT)_filename"]
    RCT_dataset =       pd.read_excel(datadir + RCT_dataset_filename)

    # Overall data______________________________________________________________
    RCTxLIG_dataset_filename = trial_setup_dict["dataset(RCTxLIG)_filename"]
    RCTxLIG_dataset =   pd.read_excel(datadir + RCTxLIG_dataset_filename)

    if unique_num_dict==None:
        unique_num_dict = { 'lig':17, 'rct':20}

    # get settings of data---------------------------------------------------------------------------------------------
    
    # decide use what data
    input_ndata_list = trial_setup_dict["input_ndata_list"]
    input_edata_list = trial_setup_dict["input_edata_list"]

    #???
    constant_node_num = trial_setup_dict.get("constant_node_num")

    # treating RCTxLIG------------------------------------------------------------------------------------------------------------------
    
    RCTxLIG_dataset.pop('rct')
    RCTxLIG_dataset.pop('lig')

    

    # gen train set
    # if specified lig or rct ids
    if vali_tag != None or test_tag != None:

        if not vali_tag:
            vali_tag = ["~~NO_SUCH_TAG~~"] # for convenience

        vali_RCTxLIG_dataset =RCTxLIG_dataset.loc[RCTxLIG_dataset['label'].isin(vali_tag)]
        test_RCTxLIG_dataset =RCTxLIG_dataset.loc[RCTxLIG_dataset['label'].isin(test_tag)]

        train_RCTxLIG_dataset = RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)
        train_RCTxLIG_dataset = train_RCTxLIG_dataset.drop(test_RCTxLIG_dataset.index)

    elif _id_dict != _EMPTY_id_dict: 

        print("selecting specific rct and lig, vali_split and test_split is ignored")

        for _y in ['rct','lig']:
            if _id_dict['train'][_y] == []:
                _id_dict['train'][_y] = [str(i) for i in range(1,unique_num_dict[_y]+1)]
                for _id in _id_dict['vali'][_y]:
                    _id_dict['train'][_y].remove(_id)
                for _id in _id_dict['test'][_y]:
                    _id_dict['train'][_y].remove(_id)

        train_tag = [ "{}_{}".format(rct_id,lig_id) for rct_id in _id_dict['train']['rct'] for lig_id in _id_dict['train']['lig']]
        
        train_RCTxLIG_dataset =RCTxLIG_dataset.loc[RCTxLIG_dataset['label'].isin(train_tag)].sample(frac=1, random_state=0)

        testvali_RCTxLIG_dataset = RCTxLIG_dataset.drop(train_RCTxLIG_dataset.index)
        
        # create empty dataframe
        _empty = {}
        for _col in RCTxLIG_dataset.columns:
            _empty[_col] = None
        vali_RCTxLIG_dataset = pd.DataFrame([_empty])
        for _y in ["rct","lig"]:
            # at least one of them must be empty
            if _id_dict['vali'][_y] != []:
                _id_list = [ int(_x) for _x in _id_dict['vali'][_y] ]
                vali_RCTxLIG_dataset = testvali_RCTxLIG_dataset.loc[testvali_RCTxLIG_dataset['{}_id'.format(_y)].isin(_id_list)]
                testvali_RCTxLIG_dataset = testvali_RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)
        test_RCTxLIG_dataset = pd.DataFrame([_empty])
        for _y in ["rct","lig"]:
            # at least one of them must be empty
            if _id_dict['test'][_y] != []:
                _id_list = [ int(_x) for _x in _id_dict['test'][_y] ]
                test_RCTxLIG_dataset = testvali_RCTxLIG_dataset.loc[testvali_RCTxLIG_dataset['{}_id'.format(_y)].isin(_id_list)]

                testvali_RCTxLIG_dataset = testvali_RCTxLIG_dataset.drop(test_RCTxLIG_dataset.index)
        # fill remaining to valid if no requirement
        if vali_RCTxLIG_dataset['label'].to_list() == [None] and _id_dict['vali'] == {"rct":[],"lig":[]}:
            vali_RCTxLIG_dataset = train_RCTxLIG_dataset.sample(frac=vali_split, random_state=0)
            train_RCTxLIG_dataset = train_RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)
        
        # fill remaining to test if no requirement
        if test_RCTxLIG_dataset['label'].to_list() == [None] and _id_dict['test'] == {"rct":[],"lig":[]}:
            test_RCTxLIG_dataset = train_RCTxLIG_dataset.sample(frac=test_split, random_state=0)
            train_RCTxLIG_dataset = train_RCTxLIG_dataset.drop(test_RCTxLIG_dataset.index)
        
        if list(testvali_RCTxLIG_dataset.index) != []:
            print("ERROR, there is something wrong with the valid, test set setting, pls change it")
            exit()

    # if have test split
    else:
        nontrain_frac = vali_split + test_split
        train_RCTxLIG_dataset = RCTxLIG_dataset.sample(frac=1-nontrain_frac, random_state=0)

        # gen test set
        vali_test_RCTxLIG_dataset = RCTxLIG_dataset.drop(train_RCTxLIG_dataset.index)
        vali_frac = vali_split/(vali_split+test_split)
        vali_RCTxLIG_dataset = vali_test_RCTxLIG_dataset.sample(frac=vali_frac, random_state=0)
        test_RCTxLIG_dataset = vali_test_RCTxLIG_dataset.drop(vali_RCTxLIG_dataset.index)


    

    # for info
    train_vali_test_indexes = {
        'train_index':list(train_RCTxLIG_dataset.index),
        'vali_index' :list(vali_RCTxLIG_dataset.index),
        'test_index' :list(test_RCTxLIG_dataset.index),
        'train_labels':train_RCTxLIG_dataset['label'].to_list(),
        'vali_labels':vali_RCTxLIG_dataset['label'].to_list(),
        'test_labels':test_RCTxLIG_dataset['label'].to_list(),
    }
    vali_RCTxLIG_dataset.reset_index(drop=True, inplace=True)
    test_RCTxLIG_dataset.reset_index(drop=True, inplace=True)


    train_tag = train_RCTxLIG_dataset.pop('label')
    train_values = train_RCTxLIG_dataset.pop("k")
    stage1_train_values = train_RCTxLIG_dataset.pop("delta_E")
    
    vali_tag = vali_RCTxLIG_dataset.pop('label')
    vali_values = vali_RCTxLIG_dataset.pop("k")
    stage1_vali_values = vali_RCTxLIG_dataset.pop("delta_E")

    test_tag = test_RCTxLIG_dataset.pop('label')
    test_values = test_RCTxLIG_dataset.pop("k")
    stage1_test_values = test_RCTxLIG_dataset.pop("delta_E")

    if as_dataset == True:
        _mode = "train"
    else:
        _mode = "vali_test"
        batch_size = 1e10


    
    # ________________________________________________________________________________________________
    def LIGcsv2graph(ndata,ndataname_list,edata,edataname_list,batch_size):
        graph_all = []
        node_num_list = []
        # For the edges, first group the table by graph IDs.
        edges_group = edata.groupby('oid')
        nodes_group = ndata.groupby('oid')

        # For each graph ID...
        for _OID in edges_group.groups:
            g = {}

            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges_group.get_group(_OID)
            nodes_of_id = nodes_group.get_group(_OID)
            
            node_num = len(nodes_of_id)
            node_num_list.append(node_num)
            
            # Create a graph and add it to the list of graphs and labels.
            g['src'] = edges_of_id['src'].values
            g['dst'] = edges_of_id['dst'].values

            # add feature of edges
            g['h_e'] = edges_of_id[edataname_list].values
            g['h_n'] = nodes_of_id[ndataname_list].values
            g['id'] = nodes_of_id['id'].values

            graph_all.append(g)

        OID_list = list(edges_group.groups.keys())

        return graph_all , OID_list, node_num_list

    if LIG_data_mode != "extraONLY":
        LIG_dataset , LIG_OID_list, node_num_list = LIGcsv2graph(LIG_NODEdataset,input_ndata_list,LIG_EDGEdataset,input_edata_list,batch_size)
    else:
        LIG_dataset = None
        node_num_list= None

    # ________________________________________________________________________________________________
    # treating RCT
    RCT_dataset.pop("rct")
    RCT_OID_list = RCT_dataset.pop("OID")
    RCT_dataset = RCT_dataset.values.tolist()

    # treating LIG extra
    if LIG_EXTRAdataset is not None:
        LIG_EXTRAdataset.pop("lig")
        LIG_EXTRA_OID_list = LIG_EXTRAdataset.pop("OID")
        LIG_EXTRAdataset = LIG_EXTRAdataset.values.tolist()

    def dictlist2graph(g_list):
        node_label_len = g_list[0]['h_n'].shape[0]
        g_all = {   'src':np.array([]),
                    'dst':np.array([]),
                    'h_e':np.array([]).reshape(0,1),
                    'h_n':np.array([]).reshape(0,1),
                    'id':np.array([]),
                }
        for i in range(len(g_list)):
            g = g_list[i]
            _corr = i * node_label_len
            g_all['src'] = np.concatenate( (g_all['src'] ,  g['src'] + _corr  ) , axis=0)
            g_all['dst'] = np.concatenate( (g_all['dst'] ,  g['dst'] + _corr  ) , axis=0)
            g_all['h_e'] = np.concatenate( (g_all['h_e'] ,  g['h_e']          ) , axis=0)
            g_all['h_n'] = np.concatenate( (g_all['h_n'] ,  g['h_n']          ) , axis=0)
            g_all['id']  = np.concatenate( (g_all['id']  ,  g['id']           ) , axis=0)

        src = tf.convert_to_tensor(g_all['src'],dtype=np.int32)
        dst = tf.convert_to_tensor(g_all['dst'],dtype=np.int32)

        node_num = (len(g_list))*node_label_len
        
        _graph = dgl.graph((src, dst), num_nodes=node_num)
        _graph.edata['h_e'] = tf.convert_to_tensor(g_all['h_e'],dtype=np.float32)
        _graph.ndata['h_n'] = tf.convert_to_tensor(g_all['h_n'],dtype=np.float32)
        _graph.ndata['id'] = tf.convert_to_tensor(g_all['id'])

        return _graph

    def get_ligbatchinput(lig_tmpbucket, ligxtr_tmpbucket, LIG_data_mode):
        # only include graphical data for ligand
        if LIG_data_mode == "graphONLY":
            return dictlist2graph(lig_tmpbucket)  # turn the list of dict to 1 graph
        # only include extra data for ligand
        elif LIG_data_mode == "extraONLY":
            return ligxtr_tmpbucket
        # include both graph and extra data 
        elif LIG_data_mode == "graphANDextra":
            return [ dictlist2graph(lig_tmpbucket) , ligxtr_tmpbucket ]  # turn the list of dict to 1 graph
        else:
            print("NO such data mode for LIG_data_mode in get_batcheddataset() :",LIG_data_mode)
            exit()

    def get_batcheddataset(input_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,input_values,LIG_data_mode):
        output_dataset = []
        batch_bucket = [ [ [],[] ],[] ]
        lig_tmpbucket = []
        ligxtr_tmpbucket = []
        count = 0
        values_list = input_values.values.tolist()
        total_count = len(input_dataset.values.tolist())
        for rct_id, lig_id in input_dataset.values.tolist():
            if rct_id == None or lig_id == None:
                break
            # rct_id, lig_id
            rct_input = RCT_dataset[rct_id-1]

            if LIG_dataset is not None:
                lig_input = LIG_dataset[lig_id-1]
                lig_tmpbucket.append(lig_input)
            if LIG_EXTRAdataset is not None:
                ligxtr_input = LIG_EXTRAdataset[lig_id-1]
                ligxtr_tmpbucket.append(ligxtr_input)

            value = values_list[count]

            batch_bucket[0][0].append( rct_input )
            #batch_bucket[0][1].append( lig_input ) #OLD code
            batch_bucket[1].append(value)

            if (count+1)%batch_size == 0 and (count+1+batch_size)<=total_count+1:
                batch_bucket[0][1] = get_ligbatchinput(lig_tmpbucket,ligxtr_tmpbucket,LIG_data_mode)
                output_dataset.append(batch_bucket)
                lig_tmpbucket = []
                ligxtr_tmpbucket = []
                batch_bucket = [ [ [],[] ],[] ]
            count += 1
        if batch_bucket != [ [ [],[] ],[] ]: #and lig_tmpbucket != []
            batch_bucket[0][1] = get_ligbatchinput(lig_tmpbucket,ligxtr_tmpbucket,LIG_data_mode)
            output_dataset.append(batch_bucket)

        if output_dataset == []:
            return [ (None,None) ]
        return output_dataset

    train_dataset = get_batcheddataset(train_RCTxLIG_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,train_values,LIG_data_mode)
    stage1_train_dataset = get_batcheddataset(train_RCTxLIG_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,stage1_train_values,LIG_data_mode)

    vali_dataset = get_batcheddataset(vali_RCTxLIG_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,vali_values,LIG_data_mode)
    stage1_vali_dataset = get_batcheddataset(vali_RCTxLIG_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,stage1_vali_values,LIG_data_mode)

    test_dataset = get_batcheddataset(test_RCTxLIG_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,test_values,LIG_data_mode)
    stage1_test_dataset = get_batcheddataset(test_RCTxLIG_dataset,RCT_dataset,LIG_dataset,LIG_EXTRAdataset,stage1_test_values,LIG_data_mode)


    
    if constant_node_num == True:
        node_num_info = node_num_list[0]
    else:
        node_num_info = None    
            
    _output_trainmode = {
        "stage1_train_dataset": stage1_train_dataset,
        "train_dataset": train_dataset,

        "stage1_vali_dataset": stage1_vali_dataset,
        "vali_dataset": vali_dataset,

        "stage1_test_dataset": stage1_test_dataset,
        "test_dataset": test_dataset,

        "node_num_info": node_num_info,
    }
    _output_valitestmode = {
        "train_tag": train_tag,
        "train_features": train_dataset[0][0],
        "train_values": train_dataset[0][1],

        "vali_tag": vali_tag,
        "vali_features": vali_dataset[0][0],
        "vali_values": vali_dataset[0][1],

        "test_tag": test_tag,
        "test_features": test_dataset[0][0],
        "test_values": test_dataset[0][1],
    }

    

    if _mode == "train":
        _output = _output_trainmode #list(_output_trainmode.values())
    elif _mode == "vali_test":
        _output = _output_valitestmode  #list(_output_valitestmode.values())


    if return_indexes:
        _output[ "train_vali_test_indexes" ] = train_vali_test_indexes

    return _output
