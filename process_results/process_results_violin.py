import numpy as np;
import pickle;
import os;
import sys;
import matplotlib.pyplot as plt; 
import seaborn as sns



def process_results_hist(in_dir, out_dir, model_prefix, dataset_name, threshold):

    result_files_prefix = os.path.join(in_dir, model_prefix);
    out_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy');
    pred = np.load(result_files_prefix + '_pred_new.npy');

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(l==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred= pred.squeeze();
    pred = pred[:,:,1] ;
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    # Calculate the histogram of the pos count in each label category
    hist1 = np.histogram(pred_n1, bins=np.arange(0,65, 5))
    hist1[0].dump(out_files_prefix + '_' + dataset_name + '_hist1y_step5.npy');
    hist1[1].dump(out_files_prefix + '_' + dataset_name + '_hist1x_step5.npy');
    hist2 = np.histogram(pred_n2, bins=np.arange(0,65, 5))
    hist2[0].dump(out_files_prefix + '_' + dataset_name + '_hist2y_step5.npy');
    hist2[1].dump(out_files_prefix + '_' + dataset_name + '_hist2x_step5.npy');
    hist3 = np.histogram(pred_n3, bins=np.arange(0,65, 5))
    hist3[0].dump(out_files_prefix + '_' + dataset_name + '_hist3y_step5.npy');
    hist3[1].dump(out_files_prefix + '_' + dataset_name + '_hist3x_step5.npy');

    # Visualize the histograms
    for i in range(1,4):
        histy = np.load(out_files_prefix + '_' + dataset_name + '_hist'+str(i)+'y_step5.npy')
        histx = np.load(out_files_prefix + '_' + dataset_name + '_hist'+str(i)+'x_step5.npy')
        plt.bar(histx_s5[1:], histy_s5)
        #plt.plot(histx_s5[1:], histy_s5, label="inc")
        plt.plot(histx_s5[1:], histy_s5)
        plt.legend()
        plt.xticks(np.arange(0,histx_s5[-1]+1,5))
        plt.show();

    return;


def process_results_violin(in_dir, out_dir, model_prefix, dataset_name, threshold, plot_type=1, exclude_ctype=None):

    result_files_prefix = os.path.join(in_dir, model_prefix );
    out_files_prefix = os.path.join(in_dir, model_prefix + '_'+dataset_name);
    lbl = np.load(result_files_prefix + '_individual_labels.npy');
    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy');
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy');    
    #print('pred.shape = ', pred.shape)
    pred= pred.squeeze();
    #print('pred.shape = ', pred.shape)
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    if(not (exclude_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred = pred[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    #print(pred_n1) ;
    #print(np.where(lbl2 == 1)) ;
    #print(lbl[np.where(lbl2 == 1)]) ;

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #data = {'pred_n':pred_n}
    #sns.violinplot(y=pred_n3, bw=1) # multiplies bw by the std to control smoothness
    #sns.violinplot(y=pred_n3, bw=1, cut=0) # cut =0 means do not extend beyond data range default is 2
    #sns.violinplot(y=pred_n3, bw=1, cut=0, scale='count') # scale reflects the relative shapes of the different violins 'width:same width, area:same area, count:width relative to count in category'
    #sns.violinplot(y=pred_n3, bw=1, cut=0, scale='width')
    #sns.violinplot(y=pred_n3, bw=1, cut=0, width=0.5) # the width of the violin default is 0.8
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig(model_prefix +'_' +dataset_name+'_violin.png');
    return;

def save_n_pred_pos(in_dir, model_prefix, threshold):

    result_files_prefix = os.path.join(in_dir, model_prefix );
    #pred = np.load(result_files_prefix + '_pred_new.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy');
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy');    

    # get the sub patches that are predicted positive according to threshold
    # the pred is super patch -> sub patch -> logit neg, logit pos
    pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    if(len(pred.shape) > 2 and pred.shape[2]>1):
        pred = pred[:,:,1];
    elif(len(pred.shape) > 2 and pred.shape[2]==1):
        pred = pred[:,:,0];
    elif(len(pred.shape) == 2):
        pred = pred;
    pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    pred_n = pred_b.sum(axis = 1)

    pred_n.dump(result_files_prefix + '_pred_n.npy');
  
    return;

def process_results_violin_old_model(in_dir, out_dir, model_prefix, dataset_name, plot_type=1, exclude_ctype=None):

    result_files_prefix = os.path.join(in_dir, model_prefix);
    out_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy');
    pred_n_str = np.load(result_files_prefix + '_pred_old.npy');
    if(not (exclude_ctype is None)):
        ctype = pickle.load(open(result_files_prefix + '_cancer_type.pkl', 'rb'));
        ctype = np.array(ctype);
        pred_n_str = pred_n_str[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    ## get the sub patches that are predicted positive according to threshold
    ## the pred is super patch -> sub patch -> logit neg, logit pos
    #pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    #pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    #pred_n = pred_b.sum(axis = 1)
    pred_n = pred_n_str.astype(np.int)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=0.5, cut=0, width=0.5, scale='width', ax=ax)
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.5, scale='width', ax=ax)
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig('baseline.png');
    return;

def process_results_violin_old_model_w_thresh26(csv_path, plot_type=1, exclude_ctype=None):

    cancer_type_list = [];
    filename_list = [];
    individual_labels_list = []
    avg_label_list = []
    pred_old_list = []

    # read csv file
    with open(csv_path, 'r') as label_file:
        line = label_file.readline(); # skip title line
        line = label_file.readline();
        while(line):
            c, s, p, i1, i2, i3, i4, i5, i6, pred_old, pred_thresh23_nec, pred_thresh23, pred_thresh26_nec, pred_thresh26= line.split(',');
            if (i1.strip()==""):
                i1 = 0;
            if (i2.strip()==""):
                i2 = 0;
            if (i3.strip()==""):
                i3 = 0;
            if (i4.strip()==""):
                i4 = 0;
            if (i5.strip()==""):
                i5 = 0;
            if (i6.strip()==""):
                i6 = 0;
            cancer_type_list.append(c);
            filename_list.append(s+'_'+p+'.png');
            individual_labels_list.append([int(i1), int(i2), int(i3), int(i4), int(i5), int(i6)]);
            avg_label_list.append(np.mean(np.array([float(i1), float(i2), float(i3), float(i4), float(i5), float(i6)])));
            #pred_old_list.append(pred_old);
            pred_old_list.append(pred_thresh26);
            line = label_file.readline();

    lbl = np.array(individual_labels_list);
    pred_n_str = np.array(pred_old_list);
    if(not (exclude_ctype is None)):
        ctype = np.array(cancer_type_list);
        pred_n_str = pred_n_str[np.where(ctype!=exclude_ctype)]
        lbl = lbl[np.where(ctype!=exclude_ctype)]

    # label values are: 1, 2, 3, 4,; empty,  ignore value of 4 and empty
    lbl[np.where(lbl==4)] = 0   ;
    # to get the average score label need to get the count of scores available for each super patch
    b = lbl>0;
    n = b.sum(axis = 1);
    n[np.where(n ==0)] = -1; # set to -1 the count = 0 to avoid division by zero
    # get the average score label by summing each patch scores and divide by count then round
    lbl2 = np.divide(lbl.sum(axis = 1), n);
    lbl2 = np.round(lbl2);

    ## get the sub patches that are predicted positive according to threshold
    ## the pred is super patch -> sub patch -> logit neg, logit pos
    #pred= pred.squeeze();
    #pred = pred[:,:,1] ;
    #pred_b = pred > threshold ;

    # get the number of subpatches predicted positive in each superpatch
    #pred_n = pred_b.sum(axis = 1)
    pred_n = pred_n_str.astype(np.int)
    # get the number of subpatches predicted positive in each superpatch in each score label category 1,2,3
    pred_n1 = pred_n[np.where(lbl2 == 1)]
    pred_n2 = pred_n[np.where(lbl2 == 2)]
    pred_n3 = pred_n[np.where(lbl2 == 3)]

    if(plot_type == 0 or plot_type == 1 or plot_type == 2):
        if(not(0 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [0]))

        if(not(64 in pred_n1)):
            pred_n1 = np.concatenate((pred_n1, [64]))

        if(not(0 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [0]))

        if(not(64 in pred_n2)):
            pred_n2 = np.concatenate((pred_n2, [64]))

        if(not(0 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [0]))

        if(not(64 in pred_n3)):
            pred_n3 = np.concatenate((pred_n3, [64]))

    fig,ax = plt.subplots(1)
    sns.set(style="whitegrid")
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=0.5, cut=0, width=0.5, scale='width')
    #ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.5, scale='width')
    if(plot_type == 0):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, width=0.7, scale='width', ax=ax)
    elif(plot_type == 1):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], bw=1, cut=0, width=0.5, scale='width', ax=ax) # original (1)
    elif(plot_type == 2 or plot_type == 3):
        ax = sns.violinplot(data=[pred_n1,pred_n2,pred_n3], cut=0, ax=ax)
    ax.set(xticklabels=['low', 'medium', 'high'])
    #plt.show();
    fig.savefig('baseline_th26.png');
    return;

if __name__ == "__main__":
    #sys.argv

    ##model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_e0";
    ##model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_e2";
    ##model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_semiauto_e81";
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_semiauto_e23";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/violin";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/violin";
    #dataset_name = "superpatches_categorized"
    #threshold = 0.5;
    ##threshold = 0.6;
    ##threshold = 0.7;

    in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/superpatch_new";
    out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/superpatch_new";
    dataset_name = "superpatches_merged"

    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_bytype_dk5e-1_cont2_e3" # cutoff =  0.15
    ##threshold = 0.2;
    #threshold = 0.15; # youden
    ##threshold = 0.5;
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_e0" # cutoff =  0.58
    ##threshold = 0.5;
    #threshold = 0.58; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_e2" # cutoff =  0.30
    ##threshold = 0.5;
    ###threshold = 0.2;
    #threshold = 0.3; # youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_semiauto_e81"; # cutoff =  0.52  
    ##threshold = 0.5;
    #threshold = 0.52; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_semiauto_e23"; # cutoff =  0.50
    ##threshold = 0.5;
    #threshold = 0.5; # youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_manual_e22"; # cutoff =  0.25
    ##threshold = 0.5;
    #threshold = 0.25; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_manual_e19";  # cutoff =  0.37
    ##threshold = 0.5;
    #threshold = 0.37; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_manual_e5"; # cutoff_youden =  0.42531168, cutoff_distance =  0.3960078
    ##threshold = 0.5;
    #threshold = 0.43;# youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_manual_e4"; # cutoff_youden =  0.4241887, cutoff_distance =  0.35299426
    ##threshold = 0.5;
    #threshold = 0.42;# youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_semi_e0"; # cutoff_youden = 0.6106335 cutoff_distance = 0.6106335 auc = 0.9433193297600078
    ##threshold = 0.5;
    #threshold = 0.61;# youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_semi_e0"; # cutoff_youden = 0.4943972 cutoff_distance = 0.5410269 auc = 0.9427495291902072
    ##threshold = 0.5;
    #threshold = 0.49; # youden
    ##threshold = 0.54; # distance

    ############ new ##############
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_filtered_by_testset_e1"; # cutoff_youden =  0.46521357
    ##threshold = 0.5;
    #threshold = 0.47;# youden

    ##XXXXXXXXXXXXX
    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_filtered_by_testset_e2"; # cutoff_youden =  0.56188786
    ##threshold = 0.5;
    #threshold = 0.56;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_filtered_by_testset_e5"; # cutoff_youden =  0.12534586
    ##threshold = 0.5;
    #threshold = 0.13;# youden

    #model_prefix = "tcga_vgg16_b128_CEloss_lr5e-5_crop100_val-strat_semiauto_filtered_by_testset_e6"; # cutoff_youden =  0.33104905
    ##threshold = 0.5;
    #threshold = 0.33;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_semiauto_filtered_by_testset_e3"; # cutoff_youden =  0.4810866
    ##threshold = 0.5;
    #threshold = 0.48;# youden

    #model_prefix = "tcga_vgg16_b128_CEloss_lr5e-5_crop100_val-strat_mysubset_filtered_by_testset2_e1"; # cutoff_youden =  0.4163274
    ##threshold = 0.5;
    #threshold = 0.42;# youden

    ##XXXXXXXXXXXXX
    #model_prefix = "tcga_vgg16_b128_CEloss_lr5e-5_crop100_val-strat_mysubset_filtered_by_testset_e3"; # cutoff_youden =  0.5142801
    ##threshold = 0.5;
    #threshold = 0.51;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset_filtered_by_testset_e3"; ####### threshold = 0.585;# youden
    ##threshold = 0.5;
    #threshold = 0.59;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset_filtered_by_testset2_e5"; # cutoff_youden =  0.4829271
    ##threshold = 0.5;
    #threshold = 0.48;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset_filtered_by_testset_e12"; # cutoff_youden =  0.2299685
    ##threshold = 0.5;
    #threshold = 0.23;# youden

    #### ok
    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset_filtered_by_testset2_e14"; # cutoff_youden =  0.10033797
    ##threshold = 0.5;
    #threshold = 0.1;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset_filtered_by_testset_e15"; # cutoff_youden =  0.028811734
    ##threshold = 0.5;
    #threshold = 0.03;# youden




    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset2_e1"; # cutoff_youden =  0.5500545
    ##threshold = 0.5;
    #threshold = 0.55;# youden

    #model_prefix = "tcga_incv4_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_mysubset2_e2"; # cutoff_youden =  0.40320146
    ##threshold = 0.5;
    #threshold = 0.4;# youden

    #model_prefix = "tcga_vgg16_b128_CEloss_lr5e-5_crop100_val-strat_mysubset_e2"; # cutoff_youden =  0.5456139
    ##threshold = 0.5;
    #threshold = 0.55;# youden

    #model_prefix = "menndel"; 
    #threshold = 0.5;
    ##threshold = 0.26; #youden = 0.258

    model_prefix = "menndel_v2"; 
    #threshold = 0.5;
    threshold = 0.18; #youden

    #plot_type = 0;
    #plot_type = 1;
    #plot_type = 2;
    plot_type = 3;
    #exclude_ctype = None;
    exclude_ctype = 'uvm';
    process_results_violin(in_dir, out_dir, model_prefix, dataset_name, threshold, plot_type=plot_type, exclude_ctype =exclude_ctype );
    #save_n_pred_pos(in_dir, model_prefix, threshold)

    #model_prefix = "tcga_vgg16_b128_CEloss_lr5e-5_crop100_val-strat_mysubset_e2";
    #plot_type = 3;
    ##exclude_ctype = None;
    #exclude_ctype = 'uvm';
    ##process_results_violin_old_model(in_dir, out_dir, model_prefix, dataset_name, plot_type=plot_type, exclude_ctype =exclude_ctype );

    ##csv_path = '/pylon5/ac3uump/lhou/super-patches-evaluation/super-patches-label-with-fixed-threshold-predictions.csv';
    #csv_path = '/pylon5/ac3uump/shahira/tcga/datasets/super-patches-evaluation-merged/super-patches-label-with-fixed-threshold-predictions_m.csv';
    #plot_type = 3;
    ##exclude_ctype = None;
    #exclude_ctype = 'uvm';
    #process_results_violin_old_model_w_thresh26(csv_path, plot_type, exclude_ctype =exclude_ctype );
