# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.26.2018 
# ==============================================================================
import sys;
import os;
import tensorflow as tf;
import numpy as np;

from sa_net_arch_utilities import CNNArchUtils;
#from sa_net_loss_func import CNNLossFuncHelper, CostFuncTypes;
from sa_net_arch import AbstractCNNArch;
from sa_net_cost_func import AbstractCostFunc;

class SimpleClassifierArch(AbstractCNNArch):
    def __init__(self, n_channels, n_classes, model_out_path, model_base_filename, model_restore_filename, cost_func:AbstractCostFunc, kwargs):
        args = {'input_img_width':-1, 'input_img_height':-1, 'pretrained':False, 'freeze_layers':-1, 'extra_end_layer':-1, 
            'get_features': 'False'};

        args.update(kwargs);
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.model_out_path = model_out_path;
        self.model_base_filename = model_base_filename;
        self.model_restore_filename = model_restore_filename;
        self.cost_func = cost_func;
        self.current_model_checkpoint_path = None;
        self.input_img_width = int(args['input_img_width']);
        self.input_img_height = int(args['input_img_height']);
        self.input_x = tf.placeholder(tf.float32, shape=(None, self.input_img_height, self.input_img_width, n_channels));
        self.labels = tf.placeholder(tf.float32, shape=(None, n_classes));
        self.isTest = tf.placeholder(tf.bool);
        self.isTraining = tf.placeholder(tf.bool);
        self.dropout = tf.placeholder(tf.float32);
        #self.class_weights = tf.Variable(tf.ones([n_classes]));
        self.epochs_count = tf.Variable(0);

        self.logits = self.create_model(self.input_x, self.isTest, kwargs);
        self.cost = self.cost_func.calc_cost(self.logits, self.labels);
        #self.prediction_softmax = self.get_prediction_softmax(self.logits);
        #self.prediction_class = self.get_class_prediction(self.logits);
        print('here');
        self.correct_pred = self.get_correct_prediction(self.logits, self.labels);
        self.accuracy = self.get_accuracy();
        self.saver = tf.train.Saver(max_to_keep=100000);

    def create_model(self, input_x, isTest, kwargs):
        # predefined list of arguments
        args = { 'dropout':0.75};

        
        args.update(kwargs);

        # read extra argument
        dropout = float(args['dropout']);
        keep_prob = dropout;

        conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
        conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
        conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
        conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

        # 1, 2
        conv1 = tf.nn.conv2d(input_x, conv1_filter, strides=[1,1,1,1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv1_bn = tf.layers.batch_normalization(conv1_pool)

        # 3, 4
        conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')    
        conv2_bn = tf.layers.batch_normalization(conv2_pool)
  
        # 5, 6
        conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        conv3_bn = tf.layers.batch_normalization(conv3_pool)
    
        # 7, 8
        conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
        conv4 = tf.nn.relu(conv4)
        conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        conv4_bn = tf.layers.batch_normalization(conv4_pool)
    
        # 9
        flat = tf.contrib.layers.flatten(conv4_bn)  

        # 10
        full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
        full1 = tf.nn.dropout(full1, keep_prob)
        full1 = tf.layers.batch_normalization(full1)
    
        # 11
        full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
        full2 = tf.nn.dropout(full2, keep_prob)
        full2 = tf.layers.batch_normalization(full2)
    
        # 12
        full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
        full3 = tf.nn.dropout(full3, keep_prob)
        full3 = tf.layers.batch_normalization(full3)    
    
        # 13
        full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
        full4 = tf.nn.dropout(full4, keep_prob)
        full4 = tf.layers.batch_normalization(full4)        
    
        # 14
        logits = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)
                
        return logits;

    def restore_model(self, sess):
        if(self.model_restore_filename is None):
            self.filepath = None;
            ##debug
            #print('self.model_restore_filename is None')
            return None;
        self.filepath = os.path.join(self.model_out_path, self.model_restore_filename + '.ckpt');
        ##debug
        #print('filepath =', self.filepath )
        if(not os.path.isfile(self.filepath)):
            filepath_pattern = os.path.join(self.model_out_path, self.model_base_filename + '*.ckpt');
            list_of_files = glob.glob(filepath_pattern);
            if(len(list_of_files) <= 0):
                return None;
            self.filepath = max(list_of_files);
            print(self.filepath);
            if(not os.path.isfile(self.filepath)):
                return None;

        self.saver.restore(sess, self.filepath);
