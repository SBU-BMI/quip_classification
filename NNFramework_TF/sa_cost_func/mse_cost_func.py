# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;
import numpy as np;

from sa_net_cost_func import AbstractCostFunc;
from sa_net_loss_func_helper import CNNLossFuncHelper;
from sa_net_arch_utilities import CNNArchUtils;

class MSECost(AbstractCostFunc):
    def __init__(self, n_classes, kwargs):
        # predefined list of arguments
        args = {'class_weights':None};

        args.update(kwargs);
        class_weights = args['class_weights'];
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$');
        print(class_weights);
        if(class_weights is not None):
            class_weights = np.array([float(x) for x in class_weights.split(',')])
            class_weights = class_weights.reshape((1, -1));
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$');
        print(class_weights);
        self.n_classes = n_classes;

        if(class_weights is None):
            self.class_weights = tf.Variable(tf.ones([self.n_classes]));
        else:
            self.class_weights = tf.Variable(class_weights);



    def calc_cost(self, logits, labels):
        return CNNLossFuncHelper.cost_mse(logits, labels, self.class_weights, self.n_classes);



    
