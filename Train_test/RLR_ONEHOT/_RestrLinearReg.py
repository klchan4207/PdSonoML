import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str)
parser.add_argument('--useEq', action='store_true')
parser.add_argument('--evalAll', action='store_true')
parser.add_argument('--trial_num', type=int, default=100)
_args = parser.parse_args()
_mode = _args.mode
_useEq = _args.useEq
_evalAll = _args.evalAll
_trial_num = _args.trial_num

# _mode = 'ONEHOT'
# _useEq = False
# _evalAll = False
# _trial_num = 100

data = pd.read_excel('../../DATA/_RAWDATA(RCTxLIG)_deltaE.xlsx')

lig_data1 = pd.read_excel('../../DATA/_rawDATA_BV_CA_MIN(LIG).xlsx')
lig_data2 = pd.read_excel('../../DATA/_gen_MFF_MFFdata(LIG).xlsx')
rct_data = pd.read_excel('../../DATA/_RAWDATA(RCT).xlsx')
data = data.merge(lig_data1, how='left', on='lig')
data = data.merge(lig_data2, how='left', on='lig')
data = data.merge(rct_data, how='left', on='rct')

X_rct = data[['rct_m_electronic', 'rct_p_electronic']].values

if _mode == 'BV':
    X_lig = data[['buried_volume']].values
elif _mode == 'CA':
    X_lig = data[['cone_angle']].values
elif _mode == 'BVCA':
    X_lig = data[['buried_volume','cone_angle']].values
elif _mode == 'ONEHOT':
    X_lig = OneHotEncoder().fit_transform(data[['lig']]).toarray()
elif _mode == 'MFF':
    X_lig = data[[col for col in data.columns if 'MFF' in col]].values
print('Current ligand descriptor: {}'.format(_mode))

X = np.concatenate((X_rct, X_lig), axis=1)

y = data[['delta_E','k']].values

_model_dict = {
    'train':{
        'y':[],
        'y_pred':[],
        'R^2':[],
        'MAE':[],
        'RMSE':[],
    },
    'val':{
        'y':[],
        'y_pred':[],
        'R^2':[],
        'MAE':[],
        'RMSE':[],
    },
    'test':{
        'y':[],
        'y_pred':[],
        'R^2':[],
        'MAE':[],
        'RMSE':[],
    },
}

import warnings
warnings.filterwarnings('ignore')
def main():
    for i in range(_trial_num):
        X_train, X_valtest, y_train_Ek , y_valtest_Ek  = train_test_split(X, y, test_size=.4, random_state=i)
        X_test, X_val, y_test_Ek , y_val_Ek  = train_test_split(X_valtest, y_valtest_Ek , test_size=0.5, random_state=i)
        yE_train , yk_train = np.transpose(y_train_Ek)
        _        , yk_val = np.transpose(y_val_Ek)
        _        , yk_test = np.transpose(y_test_Ek)


        model = LinearRegression(positive=True,fit_intercept=False)
            

        if _useEq:
            model.fit(X_train, yE_train)
            yk_train_pred = ArrheniusEq(model.predict(X_train).reshape(-1)).numpy()
            yk_val_pred =  ArrheniusEq(model.predict(X_val).reshape(-1)).numpy()
            yk_test_pred =  ArrheniusEq(model.predict(X_test).reshape(-1)).numpy()
        else:
            model.fit(X_train, yk_train)
            yk_train_pred = model.predict(X_train).reshape(-1)
            yk_val_pred =  model.predict(X_val).reshape(-1)
            yk_test_pred =  model.predict(X_test).reshape(-1)

        MAE = tf.keras.metrics.mean_absolute_error

        _model_dict['train']['R^2'].append(get_R2(yk_train,yk_train_pred))
        _model_dict['val']['R^2'].append(get_R2(yk_val,yk_val_pred))
        _model_dict['test']['R^2'].append(get_R2(yk_test,yk_test_pred))

        _model_dict['train']['MAE'].append(float(MAE(yk_train,yk_train_pred)))
        _model_dict['val']['MAE'].append(float(MAE(yk_val,yk_val_pred)))
        _model_dict['test']['MAE'].append(float(MAE(yk_test,yk_test_pred)))

        _model_dict['train']['RMSE'].append(mean_squared_error(yk_train,yk_train_pred, squared = False) )
        _model_dict['val']['RMSE'].append(mean_squared_error(yk_val,yk_val_pred, squared = False))
        _model_dict['test']['RMSE'].append(mean_squared_error(yk_test,yk_test_pred, squared = False))

    # top10 by val MAE
    _top10_idx_list = np.array(_model_dict['val']['MAE']).argsort()[:10]

    for k,v in _model_dict.items():
        print('{}:'.format(k))
        for subk,subv in v.items():
            if subk not in ['y','y_pred']:
                if not _evalAll:
                    output_list = [subv[_idx] for _idx in _top10_idx_list]
                else:
                    output_list = subv
                print('{0} = {1:.2f}'.format(subk,np.average(output_list)))







def get_R2(_real,_pred):
    _lin = LinearRegression(normalize=False,fit_intercept=True)
    _result = _lin.fit(_pred.reshape(-1, 1), _real)
    _R2 = _result.score(_pred.reshape(-1, 1), _real)
    return _R2

def ArrheniusEq(Ea):
    _h = 6.62607015*(10**(-34))
    _kb = 1.380649*(10**(-23))
    _J_to_kcal = 4184
    _R = 8.31
    _T = 273+80
    _1000s_to_h = 3.6

    return tf.math.multiply(_kb*_T/_h/_1000s_to_h,
                            tf.keras.activations.exponential( 
                                tf.math.multiply(
                                    Ea,-_J_to_kcal/_R/_T
                                    ) 
                                )
                            )
if __name__ == '__main__':
    main()