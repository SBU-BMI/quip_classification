# author: Shahira Abousamra <shahira.abousamra@stonybrook.edu>
# created: 12.23.2018 
# ==============================================================================
import tensorflow as tf;
import os;
from distutils.util import strtobool;
import glob;

from sa_net_train import CNNTrainer;
from sa_net_arch import AbstractCNNArch;
from sa_net_arch_utilities import CNNArchUtils;
from sa_net_optimizer import OptimizerTypes, CNNOptimizer;
from sa_net_data_provider import AbstractDataProvider;


class ClassifierTrainer(CNNTrainer):
    def __init__(self, cnn_arch:AbstractCNNArch, train_data_provider:AbstractDataProvider, validate_data_provider:AbstractDataProvider, optimizer_type, session_config, kwargs):
        # predefined list of arguments
        args = {'max_epochs':1000, 'learning_rate': 0.0005, 'batch_size':256, 'epoch_size':10, 'display_step':5, 'save_best_only':'False'};
        args.update(kwargs);

        self.cnn_arch = cnn_arch;
        #self.cost_func = cost_func;
        self.train_data_provider = train_data_provider;
        self.validate_data_provider = validate_data_provider;
        self.optimizer_type = optimizer_type;
        if(session_config == None):
            self.session_config = tf.ConfigProto();
            self.session_config.gpu_options.per_process_gpu_memory_fraction = 0.9
        else:
            self.session_config = session_config;
        self.max_epochs = int(args['max_epochs']);
        self.learning_rate = tf.Variable(float(args['learning_rate']));
        self.batch_size = int(args['batch_size']);
        self.epoch_size = int(args['epoch_size']);
        self.epoch_size_config = self.epoch_size ;
        self.display_step = int(args['display_step']);
        self.global_step = tf.Variable(0);
        self.save_best_only = bool(strtobool(args['save_best_only']));

        self.init = tf.global_variables_initializer();
        if(self.optimizer_type == OptimizerTypes.ADAM):
            self.optimizer = CNNOptimizer.adam_optimizer(self.learning_rate, self.cnn_arch.cost, self.global_step);
        self.epoch_out_filename = os.path.join(self.cnn_arch.model_out_path, self.cnn_arch.model_base_filename + '_train_epoch_out.txt');
        self.minibatch_out_filename = os.path.join(self.cnn_arch.model_out_path, self.cnn_arch.model_base_filename + '_train_minibatch_out.txt');

    def train(self, do_init, do_restore, do_load_data):
        self.epoch_out_filewriter = open(self.epoch_out_filename, 'a+' );
        self.minibatch_out_filewriter = open(self.minibatch_out_filename, 'a+' );
        best_saved_model_filename = None;
        last_saved_model_filename = None;
        with tf.Session(config=self.session_config) as sess:
            with sess.as_default():
                if(do_init):
                    sess.run(tf.global_variables_initializer());
                    #sess.run(self.init);
                if(do_restore):
                    #print('before restore');
                    self.cnn_arch.restore_model(sess);
                    #print('after restore');
                if(do_load_data):
                    self.train_data_provider.load_data();
                    if(not (self.validate_data_provider is None)):
                        self.validate_data_provider.load_data();            
                epoch_start_num = self.cnn_arch.epochs_count.eval();            

                if(self.epoch_size < 0):
                    self.epoch_size = int(self.train_data_provider.data_count / float(self.batch_size) + 0.5);
                if(not(self.validate_data_provider is None)):
                    self.validate_epoch_size = int(self.validate_data_provider.data_count / float(self.batch_size) + 0.5);

                best_validation_accuracy = 0;
                best_train_val_avg_accuracy = 0;
                current_validation_accuracy = None;
                current_train_val_avg_accuracy = None;    
                for epoch in range(epoch_start_num, self.max_epochs):
                    total_cost = 0;
                    total_correct_count = 0;
                    total_count = 0;
                    for step in range(0, self.epoch_size):
                        batch_x, batch_label = self.train_data_provider.get_next_n(self.batch_size);
                        opt, cost, correct_pred  = sess.run([self.optimizer, self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                                , feed_dict={self.cnn_arch.input_x: batch_x.eval() \
                                , self.cnn_arch.labels: batch_label.eval() \
                                , self.cnn_arch.isTest: False \
                                , self.cnn_arch.isTraining: True \
                            });
                 
                        total_cost += cost;
                        batch_correct_count = correct_pred.sum();
                        total_correct_count += batch_correct_count;
                        batch_count = batch_label.eval().shape[0];
                        total_count += batch_count;

                        if step % self.display_step == 0:
                            #self.output_minibatch_info(epoch, cost)
                            self.output_minibatch_info(epoch, step, cost, batch_correct_count, batch_count)
                                                    

                    print('mean loss = ', total_cost/float(total_count))
                    print('accuracy = ', batch_correct_count.sum()/float(total_count))

                    #self.output_epoch_info(epoch, total_cost);
                    self.output_epoch_info(epoch, total_cost, self.epoch_size, total_correct_count, total_count);
                    current_train_accuracy = total_correct_count / float(total_count);
                        
                    
            

                    # increment number of epochs
                    sess.run(tf.assign_add(self.cnn_arch.epochs_count, 1))

                    if(not(self.validate_data_provider is None)):
                        # run in test mode to ensure batch norm is calculated based on saved mean and std
                        print("Running Validation:");
                        self.write_to_file("Running Validation"
                            , self.epoch_out_filewriter);
                        self.validate_data_provider.reset();
                        if(not(self.validate_data_provider is None)):
                            self.validate_epoch_size = int(self.validate_data_provider.data_count / float(self.batch_size) + 0.5);
                        validate_total_loss = 0;
                        validate_correct_count = 0;
                        validate_count = 0;
                        for validate_step in range(0, self.validate_epoch_size):
                            validate_batch_x, validate_batch_label = self.validate_data_provider.get_next_n(self.batch_size);
                            if(validate_batch_x is None):
                                break;
                            cost, correct_pred  = sess.run([self.cnn_arch.cost, self.cnn_arch.correct_pred] \
                                , feed_dict={self.cnn_arch.input_x: validate_batch_x.eval() \
                                , self.cnn_arch.labels: validate_batch_label.eval() \
                                , self.cnn_arch.isTest: True \
                            });
                 
                            validate_total_loss += cost;
                            validate_correct_count += correct_pred.sum();
                            validate_count += validate_batch_label.eval().shape[0];
                            #break;

                        current_validation_accuracy = validate_correct_count / float(validate_count);
                        current_train_val_avg_accuracy = (current_train_accuracy + current_validation_accuracy)/2.0;
                        self.output_epoch_info(epoch, validate_total_loss, self.validate_epoch_size, validate_correct_count, validate_count);                        
                        #self.output_epoch_info(epoch, validate_total_loss, self.validate_epoch_size, validate_correct_count, validate_count);                        
                        #print('validation_accuracy = ', current_validation_accuracy);


                      #  if((not self.save_best_only)  
		                    #or (current_validation_accuracy is None) 
		                    #or (current_validation_accuracy >= best_validation_accuracy) \
		                    #or (current_train_val_avg_accuracy >= best_train_val_avg_accuracy) \
		                    #):
                        if((current_validation_accuracy is None) 
		                    or (current_validation_accuracy > best_validation_accuracy) \
		                    ):
	                        #self.cnn_arch.save_model(sess, '_epoch_' + str(epoch));
                            best_validation_accuracy = current_validation_accuracy;
                            #best_train_val_avg_accuracy = current_train_val_avg_accuracy;
                            print("Saving model:");
                            new_best_saved_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch);
                            self.delete_model_files(best_saved_model_filename);
                            best_saved_model_filename = new_best_saved_model_filename;
                        else:
                            new_saved_model_filename = self.cnn_arch.save_model(sess, self.optimizer, epoch);
                            self.delete_model_files(last_saved_model_filename);
                            last_saved_model_filename = new_saved_model_filename;

		                # permute the training data for the next epoch
                        self.train_data_provider.reset(repermute=True);
                        if(self.epoch_size_config < 0):
                            self.epoch_size = int(self.train_data_provider.data_count / float(self.batch_size) + 0.5);

                print("Optimization Finished!")


#    def output_minibatch_info(self, epoch, cost):
#        print("epoch = " + str(epoch) \
#            + ", minibatch loss= " + "{:.6f}".format(cost) \
#        );

#    def output_epoch_info(self, epoch, total_cost):
#        print("\r\nepoch = " + str(epoch) \
#            + ", total loss= " + "{:.6f}".format(total_cost) \
#        );

    def output_minibatch_info(self, epoch, batch, cost, correct_count, total_count):
        print("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
        );
        self.write_to_file("epoch = " + str(epoch) \
            + ", batch# = " + str(batch) \
            + ", minibatch loss= " + "{:.6f}".format(cost) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            , self.minibatch_out_filewriter
        );


    def output_epoch_info(self, epoch, total_cost, n_batches, correct_count, total_count):
        print("\r\nepoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
        );
        self.write_to_file("epoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            , self.epoch_out_filewriter
        );
        self.write_to_file("\r\n epoch = " + str(epoch) \
            + ", avg loss= " + "{:.6f}".format(total_cost / n_batches) \
            + ", correct count= " + "{:d}".format(correct_count) \
            + ", total count= " + "{:d}".format(total_count) \
            + ", accuracy = " + "{:.6f}".format(correct_count/float(total_count)) \
            , self.minibatch_out_filewriter
        );

    def write_to_file(self, text, filewriter):
        filewriter.write('\r\n');
        filewriter.write(text);
        filewriter.flush();

#    def print_optimizer_params(self):
#        # Print optimizer's state_dict
#        print("Optimizer's state_dict:")
#        for var_name in self.optimizer.state_dict():
#            print(var_name, "\t", self.optimizer.state_dict()[var_name])

    def delete_model_files(self, filepath):
        if(filepath is None):
            return;
        file_pattern = filepath + '*';
        files = glob.glob(file_pattern);
        for file in files: 
            os.remove(file);