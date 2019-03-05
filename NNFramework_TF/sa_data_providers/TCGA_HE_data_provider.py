from ..sa_net_data_provider import AbstractDataProvider;
from numpy import random;
import glob

#import scipy.io as spio;
import numpy as np;
import glob;
import os;
import pickle;
from distutils.util import strtobool;
import tensorflow as tf;
from skimage import io;
from skimage import transform as sktransform;
import math;
import sys;
import time;

from .data_aug import *;

class TCGAHEDataProvider(AbstractDataProvider):
    def __init__(self, is_test, filepath_data, filepath_label, n_channels, n_classes, do_preprocess, do_augment, data_var_name=None, label_var_name=None, permute=False, repeat=True, kwargs={}):
        #super(MatDataProvider, self).__init__(filepath_data, filepath_label, n_channels, n_classes);
        print('TCGAHEDataProvider********')
        args = {'input_img_height':460, 'input_img_width': 700, 'file_name_suffix':'', 'pre_resize':'False', 'postprocess':'False'};
        args.update(kwargs);    
        self.input_img_height = int(args['input_img_height']);
        self.input_img_width = int(args['input_img_width']);
        #print('self.input_img_width, self.input_img_height = ', self.input_img_width, self.input_img_height);
        self.file_name_suffix = args['file_name_suffix'];
        self.pre_resize = bool(strtobool(args['pre_resize']));
        self.do_postprocess = bool(strtobool(args['postprocess']));
        self.is_test = is_test; # note that label will be None when is_test is true
        self.filepath_data = filepath_data;
        if(filepath_label == None or filepath_label.strip() == ''):
            self.filepath_label = None ;
        else:
            self.filepath_label = filepath_label ;
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.do_preprocess = do_preprocess;
        self.do_augment = do_augment;
        self.data_var_name = data_var_name;
        self.label_var_name = label_var_name;
        self.do_permute = permute;
        self.do_repeat = repeat;
        if(do_augment):
            self.create_augmentation_map(kwargs);
        if(self.do_postprocess):
            self.read_postprocess_parameters(kwargs); 
        if(self.do_preprocess):
            self.read_preprocess_parameters(kwargs); 

        self.is_loaded = False;
        self.tmp_index = 0;




    def load_data(self):
        self.data = None;
        self.label = None;
        self.last_fetched_indx = -1;
        self.permutation = None;
        self.data_count = 0;
        self.data = None;
        self.labels = None;

        file_pattern = '*.png';
        file_pattern_full = os.path.join(self.filepath_data, '**', file_pattern);
        #print('file_pattern_full = ', file_pattern_full );
        self.data = glob.glob(file_pattern_full, recursive=True);
        self.data_count = len(self.data);

        ### Load data file
        #for batch_id in range(1, 6):
        #    filename = os.path.join(self.filepath_data, 'preprocess_batch_' + str(batch_id) + '.p')
        #    features, labels = pickle.load(open(filename, mode='rb'))
        #    if(batch_id == 1):
        #        self.data = features;
        #        self.labels = labels;
        #    else:
        #        self.data = np.append(self.data, features, axis=0);
        #        self.labels = np.append(self.labels, labels, axis=0);

        #self.data_count = self.data.shape[0];
        #print('data_count', self.data_count)
        #print(self.data.shape)


        ## Permutation
        if(self.do_permute == True):
            self.permutation = np.random.permutation(self.data_count)
        else:
            self.permutation = None;

        self.is_loaded = True;
        print('self.last_fetched_indx, self.data_count = ', self.last_fetched_indx, self.data_count)


    def reset(self, repermute=None):
        self.last_fetched_indx = -1;
        if(repermute == True):
            self.load_data();
            self.do_permute = True;
            self.permutation = np.random.permutation(self.data_count);

    def get_next_one(self):
        ## make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## get next data point and its corresponding label
        self.last_fetched_indx = (self.last_fetched_indx + 1);
        if(self.do_repeat == False):
            if (self.last_fetched_indx >= self.data_count):
                if(self.filepath_label == None):
                    return None;
                else:
                    return None, None;
        else:
            self.last_fetched_indx = self.last_fetched_indx % self.data_count;
        actual_indx = self.last_fetched_indx ;
        if(self.permutation is not None):
            actual_indx = self.permutation[self.last_fetched_indx];
        #data_point = self.data[actual_indx, :,:,:];
        #if(self.filepath_label == None):
        #    label = None;
        #else:
        #    label = self.labels[actual_indx,:];
        data_point, label = self.load_datapoint(actual_indx);

        ## process the data
        if(self.do_preprocess == True):
            data_point, label = self.preprocess(data_point, label);
    
       
        if(self.do_augment == True):
            #data_point, label = self.augment(data_point, label);
            mu =  0.6151888371;
            sigma =  0.2506813109;
            data_point = data_aug_img(data_point, mu, sigma, deterministic=self.is_test, idraw=-1, jdraw=-1);

        #data_point = tf.image.per_image_standardization(data_point);
        if(self.do_postprocess):
            data_point, label = self.postprocess(data_point, label);
            #data_point, label = tf.py_func(self.postprocess, [data_point, label], [tf.uint8, tf.int8]); 

        ## normalize [0,1]
        #data_point /= data_point.max();
        ## normalize [-1,1]        
        #data_point = tf.subtract(data_point, 0.5);
        #data_point = tf.multiply(data_point, 2.0);
        
        #print('data_point.shape = ', data_point.shape);
        #mean = np.mean(data_point, axis=(0,1), keepdims=True)
        #std = np.std(data_point, axis=(0,1), keepdims=True)
        #data_point = (data_point - mean) / std

        if(self.filepath_label == None):
            return data_point;
        else:
            return data_point, label;



    ## returns None, None if there is no more data to retrieve and repeat = false
    def get_next_n(self, n:int):
        ## validate parameters
        if(n <= 0):
            return None, None;

        ## make sure data is loaded first
        if(self.is_loaded == False):
            self.load_data();

        ## Get number of data points to retrieve        
        if(self.do_repeat == False):
            if (self.last_fetched_indx + n >= self.data_count):
                n = self.data_count - self.last_fetched_indx - 1;
                if(n <= 0):
                    return None, None;

        ## Get data shape
        data_size_x = self.input_img_width;
        data_size_y = self.input_img_height;    

        ##print(data_size_x, data_size_y);
        #data_points = tf.zeros((n, data_size_x, data_size_y, self.n_channels))
        ##print(data_points);
        #if(self.filepath_label is None):
        #    data_labels = None;
        #else:
        #    data_labels = tf.zeros((n, self.n_classes))

        datapoints_list = [];
        self.datapoints_files_list = []; # will be filled by load data point
        if(self.filepath_label is None):
            data_labels = None;
            labels_list = None;
        else:
            labels_list = [];
        
    
        for i in range(0, n):
            if(labels_list is None):
                #data_points[i] = self.get_next_one();
                datapoints_list.append(self.get_next_one());
            else:
                #data_points[i], data_labels[i] = self.get_next_one();
                data_point_tmp, data_label_tmp = self.get_next_one();
                datapoints_list.append(data_point_tmp);
                labels_list.append(data_label_tmp);

        #data_points = tf.stack(datapoints_list);
        data_points = np.stack(datapoints_list);
        if(labels_list is not None):
            #data_labels = tf.stack(labels_list);
            data_labels = np.stack(labels_list);
        #print('data_points =' + str(np.shape(data_points)))
        sys.stdout.flush();

        #if(self.do_augment == True):
        #    data_points, data_labels = self.augment_batch(data_points, data_labels);

        #if(self.do_postprocess):
        #    data_points, data_labels = self.postprocess_batch(data_points, data_labels);

        #data_points = tf.map_fn(lambda img:tf.image.per_image_standardization(img), data_points)

        return data_points, data_labels;


    def preprocess(self, data_point, label):
        data_point2 = data_point;

        if(self.pre_crop_center):
            starty = (data_point2.shape[0] - self.pre_crop_height)//2;
            startx = (data_point2.shape[1] - self.pre_crop_width)//2;
            endy = starty + self.pre_crop_height;
            endx = startx + self.pre_crop_width;
            data_point2 = data_point2[starty:endy, startx:endx, :];
        if(not(data_point2.shape[0] == self.input_img_height) or not(data_point2.shape[1] == self.input_img_width)):
            if(self.pre_resize):
                data_point2 = sktransform.resize(data_point2, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True).astype(np.uint8);
                #data_point2 = tf.image.resize_images(data_point2, [self.input_img_height, self.input_img_width]);
            elif(self.pre_center):
                diff_y = self.input_img_height - data_point2.shape[0];
                diff_x = self.input_img_width - data_point2.shape[1];
                diff_y_div2 = diff_y//2;
                diff_x_div2 = diff_x//2;
                data_point_tmp = np.zeros((self.input_img_height, self.input_img_width, data_point2.shape[2]));
                if(diff_y >= 0 and diff_x >= 0):
                    data_point_tmp[diff_y:diff_y+self.input_img_height, diff_x:diff_x+self.input_img_width, :] = data_point2;
                    data_point2 = data_point_tmp;
            ##debug
            #print('after resize: ', data_point2.shape);
        return data_point2, label;




    def load_datapoint(self, indx):
        filepath = self.data[indx];
        self.datapoints_files_list.append(filepath)
        #filename_queue = tf.train.string_input_producer([filepath]) #  list of files to read

        #reader = tf.WholeFileReader()
        #key, value = reader.read(filename_queue)

        #img = tf.image.decode_png(value) # use png or jpg decoder based on your files.
        #print('filepath=', filepath)
        img = io.imread(filepath);
        if(img.shape[2] > 3): # remove the alpha
            img = img[:,:,0:3];
        label_str = filepath[-5];
        if(label_str == '0'):
            #label = tf.convert_to_tensor([1, 0], dtype=tf.int8);
            label = np.array([1, 0], dtype=np.int8);
        elif(label_str == '1'):
            #label = tf.convert_to_tensor([0, 1], dtype=tf.int8);
            label = np.array([0, 1], dtype=np.int8);
        else:
            label = None;
        #data_point = tf.convert_to_tensor(img);
        data_point = img;
        return data_point, label;

    # prepare the mapping from allowed operations to available operations index
    def create_augmentation_map(self, kwargs={}):
        args = {'aug_flip_h': 'True', 'aug_flip_v': 'True', 'aug_flip_hv': 'True' \
            , 'aug_rot180': 'True', 'aug_rot90': 'False', 'aug_rot270': 'False', 'aug_rot_rand': 'False' \
            , 'aug_brightness': 'False', 'aug_brightness_min': -50,  'aug_brightness_max': 50 \
            , 'aug_saturation': 'False', 'aug_saturation_min': -1.5,  'aug_saturation_max': 1.5 \
            , 'aug_hue': 'False', 'aug_hue_min': -50,  'aug_hue_max': 50 \
            , 'aug_scale': 'False', 'aug_scale_min': 1.0,  'aug_scale_max': 2.0 \
            , 'aug_translate': 'False',  'aug_translate_y_min': -20, 'aug_translate_y_max': 20,  'aug_translate_x_min': -20, 'aug_translate_x_max': 20
            };
        #print(args);
        args.update(kwargs);    
        #print(args);
        self.rot_angles = [];
        self.aug_flip_h = bool(strtobool(args['aug_flip_h']));
        self.aug_flip_v = bool(strtobool(args['aug_flip_v']));
        self.aug_flip_hv = bool(strtobool(args['aug_flip_hv']));
        self.aug_rot180 = bool(strtobool(args['aug_rot180']));
        self.aug_rot90 = bool(strtobool(args['aug_rot90']));
        self.aug_rot270 = bool(strtobool(args['aug_rot270']));
        self.aug_rot_random = bool(strtobool(args['aug_rot_rand']));
        self.aug_brightness = bool(strtobool(args['aug_brightness']));
        self.aug_saturation = bool(strtobool(args['aug_saturation']));
        self.aug_hue = bool(strtobool(args['aug_hue']));
        self.aug_scale = bool(strtobool(args['aug_scale']));
        self.aug_translate = bool(strtobool(args['aug_translate']));
        '''
        map allowed operation to the following values
        0: same (none)
        1: horizontal flip
        2: vertical flip
        3: horizontal and vertical flip
        4: rotate 180
        5: rotate 90
        6: rotate 270 or -90
        7: rotate random angle
        '''
        self.aug_map = {};
        self.aug_map[0] = 0; # (same) none 
        i = 1;
        if(self.aug_flip_h):
            self.aug_map[i] = 1;
            i += 1;
        if(self.aug_flip_v):
            self.aug_map[i] = 2;
            i += 1;
        if(self.aug_flip_hv):
            self.aug_map[i] = 3;
            i += 1;
        if(self.aug_rot180):
            self.aug_map[i] = 4;
            self.rot_angles.append(math.pi);
            i += 1;
        if(self.aug_rot90):
            #print('self.aug_rot90={}'.format(self.aug_rot90));
            self.aug_map[i] = 5;
            self.rot_angles.append(math.pi/2);
            i += 1;
        if(self.aug_rot270):
            #print('self.aug_rot270={}'.format(self.aug_rot270));
            self.aug_map[i] = 6;
            self.rot_angles.append(-math.pi/2);
            i += 1;
        if(self.aug_rot_random):
            #self.aug_map[i] = 7;
            self.aug_rot_min = int(args['aug_rot_min']);
            self.aug_rot_max = int(args['aug_rot_max']);
            for r in range(self.aug_rot_min, self.aug_rot_max+1, 5):
                self.rot_angles.append(r*math.pi/180);

        if(self.aug_brightness):
        #    self.aug_map[i] = 7;
            self.aug_brightness_min = int(args['aug_brightness_min']);
            self.aug_brightness_max = int(args['aug_brightness_max']);
            #print('self.aug_brightness_max=',self.aug_brightness_max);
            sys.stdout.flush();
        #    i += 1;
        if(self.aug_saturation):
            self.aug_saturation_min = float(args['aug_saturation_min']);
            self.aug_saturation_max = float(args['aug_saturation_max']);
        if(self.aug_hue):
            self.aug_hue_min = int(args['aug_hue_min']);
            self.aug_hue_max = int(args['aug_hue_max']);
        if(self.aug_scale):
            self.aug_scale_min = float(args['aug_scale_min']);
            self.aug_scale_max = float(args['aug_scale_max']);
        if(self.aug_translate):
            self.aug_translate_y_min = int(args['aug_translate_y_min']);
            self.aug_translate_y_max = int(args['aug_translate_y_max']);
            self.aug_translate_x_min = int(args['aug_translate_x_min']);
            self.aug_translate_x_max = int(args['aug_translate_x_max']);
        #print(self.aug_map)




    def op_translate(self, input):
        #print('translate - input.shape = ', input.shape);
        translate_y = np.random.randint(self.aug_translate_y_min, high = self.aug_translate_y_max);
        translate_x = np.random.randint(self.aug_translate_x_min, high = self.aug_translate_x_max);
        #print('translate - x,y = ', translate_x, translate_y);
        translate_transform = sktransform.AffineTransform(translation = (translate_x, translate_y));      
        data_point2 = sktransform.warp(input, translate_transform, preserve_range=True).astype(np.uint8); 
        #print('translate - data_point2.shape = ', data_point2.shape);
        return data_point2 ;

    def read_postprocess_parameters(self, kwargs={}):
        args = {'post_resize': 'False', 'post_crop_center': 'False',
            'post_crop_height': 128, 'post_crop_width': 128
            };
        #print(args);
        args.update(kwargs);    
        #print(args);
        self.post_resize = bool(strtobool(args['post_resize']));
        self.post_crop_center = bool(strtobool(args['post_crop_center']));
        if(self.post_crop_center ):
            self.post_crop_height = int(args['post_crop_height']);
            self.post_crop_width = int(args['post_crop_width']);
            self.post_crop_y1 = None;
            self.post_crop_x1 = None;

    def read_preprocess_parameters(self, kwargs={}):
        args = {'pre_resize': 'False', 'pre_crop_center': 'False',
            'pre_crop_height': 128, 'pre_crop_width': 128
            };
        #print(args);
        args.update(kwargs);    
        #print(args);
        self.pre_resize = bool(strtobool(args['pre_resize']));
        self.pre_crop_center = bool(strtobool(args['pre_crop_center']));
        if(self.pre_crop_center ):
            self.pre_crop_height = int(args['pre_crop_height']);
            self.pre_crop_width = int(args['pre_crop_width']);
            self.pre_crop_y1 = None;
            self.pre_crop_x1 = None;

    def postprocess(self, data_point, label):
        #data_point2 = data_point.copy();
        data_point2 = data_point;
        if(data_point2.shape[0] == 1):
            data_point2 = data_point2.reshape((data_point2.shape[1], data_point2.shape[2], data_point2.shape[3]));
        #print('crop .shape = ', data_point.shape);
        if(self.post_crop_center):
            starty = (data_point2.shape[0] - self.post_crop_height)//2;
            startx = (data_point2.shape[1] - self.post_crop_width)//2;
            endy = starty + self.post_crop_height;
            endx = startx + self.post_crop_width;
            #print('self.post_crop_height = ', self.post_crop_height);
            #print('self.post_crop_width = ', self.post_crop_width);
            #print('starty = ', starty);
            #print('endy = ', endy);
            #print('startx = ', startx);
            #print('endx = ', endx);
            if(starty < 0 or startx < 0): # in case rotated the width and height will have changed
                starty = (data_point2.shape[0] - self.post_crop_width)//2;
                startx = (data_point2.shape[1] - self.post_crop_height)//2;
                endy = starty + self.post_crop_height;
                endx = startx + self.post_crop_width;
            
            ##debug
            #print('data_point2.shape = ', data_point2.shape);
            #print('starty = ', starty);
            #print('startx = ', startx);
            data_point2 = data_point2[starty:endy, startx:endx, :];
            ##debug
            #print('data_point2.shape = ', data_point2.shape);


        #print('resize - data_point2.shape', data_point2.shape);
        if(self.post_resize):
            data_point2 = sktransform.resize(data_point2, (self.input_img_height, self.input_img_width), preserve_range=True, anti_aliasing=True).astype(np.uint8);

        return data_point2, label;




    def save_state(self, checkpoint_filepath):
        if(checkpoint_filepath is None):
            return;
        base_filename,_ = os.path.splitext(checkpoint_filepath);
        if(self.permutation is not None):
            filepath_perm = base_filename + '_perm.npy' ;
            self.permutation.dump(filepath_perm);
        if(self.data is not None):
            filepath_data = base_filename + '_dat.pkl' ;
            pickle.dump(self.data, open(filepath_data, 'wb'));
        filepath_param = base_filename + '_param.pkl' ;
        pickle.dump([self.last_fetched_indx, self.data_count], open(filepath_param, 'wb'));

    def restore_state(self, checkpoint_filepath):
        if(checkpoint_filepath is None):
            return;
        base_filename,_ = os.path.splitext(checkpoint_filepath);
        filepath_perm = base_filename + '_perm.npy' ;
        if(os.path.isfile(filepath_perm)):            
            self.permutation = np.load(filepath_perm);
        filepath_data = base_filename + '_dat.pkl' ;
        if(os.path.isfile(filepath_data)):            
            self.data = pickle.load(open(filepath_data, 'rb'));
        filepath_param = base_filename + '_param.pkl' ;
        if(os.path.isfile(filepath_param)):            
            self.last_fetched_indx, self.data_count = pickle.load(open(filepath_param, 'rb'));

        print('self.last_fetched_indx, self.data_count = ', self.last_fetched_indx, self.data_count)
        
