import tensorflow as tf
from tensorflow.keras import *
from tensorflow.python.framework.ops import convert_to_tensor
from tensorflow.python.framework.tensor_shape import dimension_at_index
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.ops.gen_math_ops import mat_mul

import os
currentdir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(1, currentdir+"/Layer")

from MPNN_layer import MPNN

# modified with layer in mind
# main model
class MODEL(tf.keras.Model):
    def __init__(self, _setup_dict):
        super(MODEL, self).__init__()

        self.MPNN1 = MPNN(  _setup_dict['MPNN'] )

        self.dense_dict = {}

        for _block in [ 'dense_E1',
                        'dense_E2',
                        'dense_output'
                        ]:
            _block_dict = _setup_dict[_block]

            units = _block_dict['units']

            activation = _block_dict['activation']

            use_bias = _block_dict['use_bias']

            kernel_regularizer = _block_dict['kernel_regularizer']
            if kernel_regularizer == 'None':
                kernel_regularizer=None
            elif type(kernel_regularizer) == dict:
                for k,v in kernel_regularizer.items():
                    _type = k
                    _value = v
                if _type == "l1":
                    kernel_regularizer = tf.keras.regularizers.L1(_value)
                elif _type == "l2":
                    kernel_regularizer = tf.keras.regularizers.L2(_value)
             

            kernel_constraint =  _block_dict['kernel_constraint']
            if kernel_constraint == 'NonNeg':
                kernel_constraint=tf.keras.constraints.NonNeg()
            elif kernel_constraint == 'None':
                kernel_constraint=None

            self.dense_dict[_block] = tf.keras.layers.Dense(
                                            units = units,
                                            activation = activation,
                                            use_bias = use_bias,
                                            kernel_regularizer = kernel_regularizer,
                                            kernel_constraint = kernel_constraint,
                                        )
                            
        self._h = 6.62607015*(10**(-34))
        self._kb = 1.380649*(10**(-23))
        self._J_to_kcal = 4184
        self._R = 8.31
        self._T = 273+80
        self._1000s_to_h = 3.6

        self.useArrheniusEq = True
        self.useVertexValue = False
        self.forAnalyze = False

    def ArrheniusEq(self,Ea):
        return tf.math.multiply(self._kb*self._T/self._h/self._1000s_to_h,
                                tf.keras.activations.exponential( 
                                    tf.math.multiply(
                                        Ea,-self._J_to_kcal/self._R/self._T
                                        ) 
                                    )
                                )

    def stopuseArrheniusEq(self):
        self.useArrheniusEq = False

    def startuseArrheniusEq(self):
        self.useArrheniusEq = True

    def stopuseVertexValue(self):
        self.useVertexValue = False

    def startuseVertexValue(self):
        self.useVertexValue = True

    def call(self,  input_data,
                    return_FCNN = False,
                    return_LayersInt=False,
                    _device=None,
                    training=None, # kept to generalize arguements with other models
                    ):

        _INT_dict = {}
        
        rct_inputs_data, lig_inputs_data = input_data
        
        # treat rct
        rct_inputs = tf.convert_to_tensor(rct_inputs_data)
        #rct_inputs = self.dense_dict['dense_rct1'](rct_inputs)

        # treat lig
        if _device != None:
            #print('_device',_device)
            lig_inputs = lig_inputs_data.to(_device)
        else:
            lig_inputs = lig_inputs_data
        lig_inputs = self.MPNN1(lig_inputs)
        _INT_dict['lig_inputs'] = lig_inputs
        # (optional) return vertex value directly
        if self.useVertexValue == True:
            return lig_inputs
        if self.forAnalyze == True:
            return rct_inputs_data,lig_inputs

        # E1
        inputs_E1 = tf.concat( [rct_inputs,lig_inputs],1 )
        inputs_E1 = self.dense_dict['dense_E1'](inputs_E1)
        _INT_dict['inputs_E1'] = inputs_E1

        #E2
        inputs_E2 = self.dense_dict['dense_E2'](lig_inputs)
        _INT_dict['inputs_E2'] = inputs_E2

        # concat E1 and E2
        inputs_E1_E2 = tf.concat( [inputs_E1,inputs_E2],1 ) #tf.concat( [inputs_E1,-inputs_E2],1 )
        outputs = self.dense_dict['dense_output'](inputs_E1_E2)

        if self.useArrheniusEq:
            outputs = self.ArrheniusEq(outputs)

        if return_FCNN:
            return outputs,  lig_inputs , self.dense_dict
        if return_LayersInt:
            return outputs,  lig_inputs, self.dense_dict, _INT_dict
        return outputs