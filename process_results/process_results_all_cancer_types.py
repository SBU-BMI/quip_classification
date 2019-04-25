import numpy as np;
import pickle;
import os;
import sys;
from sklearn import metrics 
#import matplotlib.pyplot as plt
from shutil import copyfile;
import glob;

def process_results_separate_types(in_dir, out_dir, model_prefix, threshold):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    types = ['coad', 'brca', 'read', 'luad', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['ucec'];
    out_filename = os.path.join(in_dir, model_prefix + '_ctypes_stats_th-'+ str(threshold)+'.csv');
    out_file = open(out_filename, 'w+');
    print(out_filename);
    out_file.write(model_prefix+'\n\n');
    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy');        
        #pred = np.load(result_files_prefix + '_pred_new.npy');
        if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
            pred = np.load(result_files_prefix + '_pred_new.npy');
        elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
            pred = np.load(result_files_prefix + '_pred_prob.npy');    
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        #pred_t1 = pred[:,1];
        if(len(pred.shape) > 1 and pred.shape[1]>1):
            pred_t1 = pred[:,1];
        elif(len(pred.shape) > 1 and pred.shape[1]==1):
            pred_t1 = pred[:,0];
        elif(len(pred.shape) == 1):
            pred_t1 = pred;
        count = lbl.shape[0];
        P = lbl[np.where(pred_t1 > threshold)].shape[0]
        TP = lbl[np.where(pred_t1 > threshold)].sum();
        FP = P - TP;
        N = count - P;
        FN = lbl[np.where(pred_t1 <= threshold)].sum();
        TN = N - FN;
        lbl_p = lbl[np.where(pred_t1 > threshold)];
        #print(filenames[np.where(lbl_p == 0)]);
        print("TP = ", TP);
        print("FP = ", FP);
        print("TN = ", TN);
        print("FN = ", FN);
        prec = TP/float(TP + FP);
        recall = TP/float(TP + FN);
        print("prec = ", prec);
        print("recall = ", recall);
        f1 = 2/(1/prec + 1/recall);
        accuracy = (TP+TN)/float(count);
        out_file.write(ctype + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
        out_file.write("Label Positive" + "," + str(TP) + "," + str(FN) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
        out_file.write("Label Negative" + "," + str(FP) + "," + str(TN) +'\r\n');
        TP_total += TP;
        FP_total += FP;
        TN_total += TN;
        FN_total += FN;
        T_total += TP + TN;
        total += count;

    prec = TP_total/float(TP_total + FP_total);
    recall = TP_total/float(TP_total + FN_total);
    f1 = 2/(1/prec + 1/recall);
    accuracy = (T_total)/float(total);

    out_file.write('\n\n');
    out_file.write('All' + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
    out_file.write("Label Positive" + "," + str(TP_total) + "," + str(FN_total) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
    out_file.write("Label Negative" + "," + str(FP_total) + "," + str(TN_total) +'\r\n');

    out_file.close();

def process_results_separate_types_sm(in_dir, out_dir, model_prefix, threshold):
    types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['ucec'];
    out_filename = os.path.join(in_dir, model_prefix + '_ctypes_stats_th-'+ str(threshold)+'_sm.csv');
    out_file = open(out_filename, 'w+');
    print(out_filename);
    out_file.write(model_prefix+'\n\n');
    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy');
        pred = np.load(result_files_prefix + '_pred_new.npy');
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        print('pred[0] = ', pred[0])
        pred_logits= np.log(pred/(1-pred))
        print('pred_logits[0] =', pred_logits[0])
        pred_exp = np.exp(pred_logits)
        print('pred_exp[0] = ', pred_exp[0])
        pred_exp_sum = pred_exp.sum(axis=-1).reshape(-1,1)
        print('pred_exp_sum[0] = ', pred_exp_sum[0])
        pred_sm = pred_exp / pred_exp_sum ;
        print('pred_sm[0] = ', pred_sm[0])
        pred_t1 = pred_sm[:,1];
        print('pred_t1[0] = ', pred_t1[0])
        count = lbl.shape[0];
        P = lbl[np.where(pred_t1 > threshold)].shape[0]
        TP = lbl[np.where(pred_t1 > threshold)].sum();
        FP = P - TP;
        N = count - P;
        FN = lbl[np.where(pred_t1 <= threshold)].sum();
        TN = N - FN;
        lbl_p = lbl[np.where(pred_t1 > threshold)];
        #print(filenames[np.where(lbl_p == 0)]);
        print("TP = ", TP);
        print("FP = ", FP);
        print("TN = ", TN);
        print("FN = ", FN);
        prec = TP/float(TP + FP);
        recall = TP/float(TP + FN);
        print("prec = ", prec);
        print("recall = ", recall);
        f1 = 2/(1/prec + 1/recall);
        accuracy = (TP+TN)/float(count);
        out_file.write(ctype + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
        out_file.write("Label Positive" + "," + str(TP) + "," + str(FN) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
        out_file.write("Label Negative" + "," + str(FP) + "," + str(TN) +'\r\n');
        TP_total += TP;
        FP_total += FP;
        TN_total += TN;
        FN_total += FN;
        T_total += TP + TN;
        total += count;
    prec = TP_total/float(TP_total + FP_total);
    recall = TP_total/float(TP_total + FN_total);
    f1 = 2/(1/prec + 1/recall);
    accuracy = (T_total)/float(total);

    out_file.write('\n\n');
    out_file.write('All' + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
    out_file.write("Label Positive" + "," + str(TP_total) + "," + str(FN_total) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
    out_file.write("Label Negative" + "," + str(FP_total) + "," + str(TN_total) +'\r\n');

    out_file.close();


def copy_patches_wpred_separate_types(in_dir, out_dir, model_prefix, threshold):
    types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['uvm'];
    #types = ['ucec'];

    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy');
        pred = np.load(result_files_prefix + '_pred_new.npy');
        pred_logits = np.log(pred / (1-pred + 0.0001))
        exp_pred = np.exp(pred_logits - np.max(pred_logits, axis=-1, keepdims=True) + 1)
        exp_pred = exp_pred[..., -1:] / np.sum(exp_pred, axis=-1, keepdims=True)
        #pred_pos = (exp_pred > threshold).squeeze();
        #print(pred_pos.shape)

        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        pred_t1 = pred[:,1];
        count = lbl.shape[0];
        pred_pos = pred_t1 > threshold;
        lbl_pos = lbl == 1;
        true_pred_files = filenames[np.where(pred_pos == lbl_pos)];
        false_pred_files = filenames[np.where(pred_pos != lbl_pos)];
        print('threshold = ', threshold);
        print('T = ', true_pred_files.shape[0])
        print('F = ', false_pred_files.shape[0])
        print('Accuracy = ', true_pred_files.shape[0] / float(true_pred_files.shape[0] +false_pred_files.shape[0]));
        for i in range(filenames.shape[0]):            
            print("----------------------------------------------");
            file = filenames[i];
            tag = 'F';
            if(pred_pos[i] == lbl_pos[i]):
                tag = 'T';
            print(file);
            file = glob.glob(file.split('.')[0] + '*')[0]           
            print(file);
            print('pred[i] = ', pred[i])
            print('pred_pos[i] = ', pred_pos[i])
            print('exp_pred[i] = ', exp_pred[i])
            print('lbl[i] = ', lbl[i])
            filename, ext = os.path.splitext(os.path.split(file)[1]);            
            dst = os.path.join(out_dir, ctype + '_' + filename + '_'+str(pred[i][0])+'-'+str(pred[i][1])+ '_' + tag + ext);
            print(dst);
            copyfile(file, dst);

        print('threshold = ', threshold);
        print('T = ', true_pred_files.shape[0])
        print('F = ', false_pred_files.shape[0])
        print('Accuracy = ', true_pred_files.shape[0] / float(true_pred_files.shape[0] +false_pred_files.shape[0]));
        #for file in true_pred_files:
        #    print(file);
        #    file = glob.glob(file.split('.')[0] + '*')[0]           
        #    filename, ext = os.path.splitext(os.path.split(file)[1]);            
        #    dst = os.path.join(out_dir, ctype + '_' + filename + '_' + 'T' + ext);
        #    copyfile(file, dst);

        #for file in false_pred_files:
        #    print(file);
        #    file = glob.glob(file.split('.')[0] + '*')[0]          
        #    filename, ext = os.path.splitext(os.path.split(file)[1]);            
        #    dst = os.path.join(out_dir, ctype + '_' + filename + '_' + 'F' + ext);
        #    copyfile(file, dst);


def process_results(in_dir, out_dir, model_prefix, dataset_name, threshold):
    out_filename = os.path.join(in_dir, model_prefix + '_'+dataset_name+'_stats_th-'+ str(threshold)+'.csv');
    out_file = open(out_filename, 'w+');

    out_file.write(model_prefix+'\n\n');
    result_files_prefix = os.path.join(in_dir, model_prefix);
    lbl = np.load(result_files_prefix + '_individual_labels.npy');
    if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
        pred = np.load(result_files_prefix + '_pred_new.npy');
    elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
        pred = np.load(result_files_prefix + '_pred_prob.npy');    
    pred = pred.squeeze();    
    print('pred.shape = ', pred.shape);
    if(len(pred.shape) > 1 and pred.shape[1]>1):
        pred_t1 = pred[:,1];
    elif(len(pred.shape) > 1 and pred.shape[1]==1):
        pred_t1 = pred[:,0];
    elif(len(pred.shape) == 1):
        pred_t1 = pred;
    count = lbl.shape[0];
    P = lbl[np.where(pred_t1 > threshold)].shape[0]
    TP = lbl[np.where(pred_t1 > threshold)].sum();
    FP = P - TP;
    N = count - P;
    FN = lbl[np.where(pred_t1 <= threshold)].sum();
    TN = N - FN;
    prec = TP/float(TP + FP);
    recall = TP/float(TP + FN);
    f1 = 2/(1/prec + 1/recall);
    accuracy = (TP+TN)/float(count);

    fpr, tpr, thresholds = metrics.roc_curve(lbl, pred_t1, pos_label=1)
    auc = metrics.auc(fpr, tpr);
    youden_index = tpr-fpr;
    cutoff_youden = thresholds[np.argmax(youden_index)]
    distance = np.sqrt(np.square(1 - tpr) + np.square(fpr));
    cutoff_distance = thresholds[np.argmin(distance)]

    out_file.write(dataset_name + ',' + "Pred. Pos." + "," + "Pred. Neg." + "," + "Prec."+ "," + "Recall" + "," + "F1" + "," + "Accuracy" + "," + "AUC" +'\r\n');
    out_file.write("Label Pos." + "," + str(TP) + "," + str(FN) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall)  + "," + "{:.2f}".format(f1) + "," + "{:.4f}".format(accuracy)+ "," + "{:.4f}".format(auc) +'\r\n');
    out_file.write("Label Neg." + "," + str(FP) + "," + str(TN) +'\r\n');


    out_file.close();

    print('auc = ', auc);
    print('cutoff_youden = ', cutoff_youden);
    print('cutoff_distance = ', cutoff_distance);

    #for i in range(thresholds.shape[0]):
    #    print('fpr = ', fpr[i], 'tpr = ', tpr[i], 'thresholds = ', thresholds[i]);

    #plt.plot(fpr, tpr);
    #plt.show();


def process_results_separate_types_no_write(in_dir, out_dir, model_prefix, threshold):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    types = ['ucec'];

    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy');
        pred = np.load(result_files_prefix + '_pred_new.npy');
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        pred_t1 = pred[:,1];
        count = lbl.shape[0];
        P = lbl[np.where(pred_t1 > threshold)].shape[0]
        TP = lbl[np.where(pred_t1 > threshold)].sum();
        FP = P - TP;
        N = count - P;
        FN = lbl[np.where(pred_t1 <= threshold)].sum();
        TN = N - FN;
        lbl_p = lbl[np.where(pred_t1 > threshold)];
        filenames_p = filenames[np.where(pred_t1 > threshold)];
        pred_t1_p = pred_t1[np.where(pred_t1 > threshold)];
        print(filenames_p[np.where(lbl_p == 0)]);
        print(pred_t1_p[np.where(lbl_p == 0)]);
        print("TP = ", TP);
        print("FP = ", FP);
        print("TN = ", TN);
        print("FN = ", FN);
        prec = TP/float(TP + FP);
        recall = TP/float(TP + FN);
        print("prec = ", prec);
        print("recall = ", recall);
        f1 = 2/(1/prec + 1/recall);
        TP_total += TP;
        FP_total += FP;
        TN_total += TN;
        FN_total += FN;

    prec = TP_total/float(TP_total + FP_total);
    recall = TP_total/float(TP_total + FN_total);
    f1 = 2/(1/prec + 1/recall);

def process_results_separate_types_auc(in_dir, out_dir, model_prefix):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    types = ['coad', 'brca', 'read', 'luad', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    #types = ['luad'];
    #out_filename = os.path.join(in_dir, model_prefix + '_ctypes_stats_th-'+ str(threshold)+'.csv');
    #out_file = open(out_filename, 'w+');

    #out_file.write(model_prefix+'\n\n');
    TP_total = 0;
    FP_total = 0;
    TN_total = 0;
    FN_total = 0;
    T_total = 0;
    total = 0;
    pred_all_t1 = None;
    lbl_all = None;
    for ctype in types:
        print(ctype);
        result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy');
        #pred = np.load(result_files_prefix + '_pred_new.npy');
        if(os.path.isfile(result_files_prefix + '_pred_new.npy')):
            pred = np.load(result_files_prefix + '_pred_new.npy');
        elif(os.path.isfile(result_files_prefix + '_pred_prob.npy')):
            pred = np.load(result_files_prefix + '_pred_prob.npy');    
        filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames = np.array(filenames);
        pred = pred.squeeze();
        #pred_t1 = pred[:,1];
        if(len(pred.shape) > 1 and pred.shape[1]>1):
            pred_t1 = pred[:,1];
        elif(len(pred.shape) > 1 and pred.shape[1]==1):
            pred_t1 = pred[:,0];
        elif(len(pred.shape) == 1):
            pred_t1 = pred;
        if(pred_all_t1 is None):
            pred_all_t1 = pred_t1;
            lbl_all = lbl;
        else:
            pred_all_t1 = np.concatenate((pred_all_t1, pred_t1), axis=-1);
            lbl_all = np.concatenate((lbl_all,lbl), axis=-1);
        fpr, tpr, thresholds = metrics.roc_curve(lbl, pred_t1, pos_label=1)
        auc = metrics.auc(fpr, tpr);
        print('auc = ', auc);
        print(' ');
        

    fpr, tpr, thresholds = metrics.roc_curve(lbl_all, pred_all_t1, pos_label=1)
    auc = metrics.auc(fpr, tpr);
    youden_index = tpr-fpr;
    cutoff = thresholds[np.argmax(youden_index)]

    #for i in range(thresholds.shape[0]):
    #    print('fpr = ', fpr[i], 'tpr = ', tpr[i], 'thresholds = ', thresholds[i]);
    print('all auc = ', auc);
    print('all cutoff = ', cutoff);

    #plt.plot(fpr, tpr);
    #plt.show();

    #out_file.write('\n\n');
    #out_file.write('All' + ',' + "Pred. Positive" + "," + "Pred. Negative" + "," + "Prec."+ "," + "Recall"+ "," + "F1" + "," + "accuracy" +'\r\n');
    #out_file.write("Label Positive" + "," + str(TP_total) + "," + str(FN_total) + "," + "{:.2f}".format(prec) + "," + "{:.2f}".format(recall) + "," + "{:.2f}".format(f1) + "," + "{:.2f}".format(accuracy) +'\r\n');
    #out_file.write("Label Negative" + "," + str(FP_total) + "," + str(TN_total) +'\r\n');

    #out_file.close();

    #fig,ax = plt.subplots(1)
    #ax.plot(fpr, tpr);
    #fig.savefig('test.png');


def process_results_separate_types_roc_curve(in_dir, model_prefix_list, label_list):
    types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];

    fig,ax = plt.subplots(1)
    for i in range(len(model_prefix_list)):
        model_prefix = model_prefix_list[i];
        label = label_list[i];
        print(model_prefix);
        TP_total = 0;
        FP_total = 0;
        TN_total = 0;
        FN_total = 0;
        T_total = 0;
        total = 0;
        pred_all_t1 = None;
        lbl_all = None;
        for ctype in types:
            print(ctype);
            result_files_prefix = os.path.join(in_dir, ctype, model_prefix);
            lbl = np.load(result_files_prefix + '_individual_labels.npy');
            pred = np.load(result_files_prefix + '_pred_new.npy');
            filenames = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
            filenames = np.array(filenames);
            pred = pred.squeeze();
            pred_t1 = pred[:,1];
            if(pred_all_t1 is None):
                pred_all_t1 = pred_t1;
                lbl_all = lbl;
            else:
                pred_all_t1 = np.concatenate((pred_all_t1, pred_t1), axis=-1);
                lbl_all = np.concatenate((lbl_all,lbl), axis=-1);
        

        fpr, tpr, thresholds = metrics.roc_curve(lbl_all, pred_all_t1, pos_label=1)

        ax.plot(fpr, tpr, label=label);

    ax.legend();
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('ROC');
    fig.savefig('test.png');

def generate_result_files_all_types_for_old_model(in_dir, out_dir, model_prefix):
    #types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];
    types = ['coad', 'brca', 'read', 'luad', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];

    for ctype in types:
        print(ctype);
        files_prefix = os.path.join(in_dir, ctype, '**', '*.png');
        result_files_prefix = os.path.join(out_dir, ctype, model_prefix);
        files = glob.glob(files_prefix, recursive=True);
        labels = [];
        pred = [];
        pred_manual_thesh = [];
        for file in files:
            filename = os.path.split(file)[1];
            label_str = filename[-5];
            label = int(label_str);
            labels.append(label);
            filename_split = filename[0:-6].split('-');
            pred_manual_thresh_str = filename_split[-1];
            pred_prob_str = filename_split[-3];
            pred_manual_thesh.append(np.array([0, int(pred_manual_thresh_str)]));
            pred.append(np.array([0, float(pred_prob_str)]));
            print(file)
            print(label)
            print(float(pred_prob_str))
            print(int(pred_manual_thresh_str))

        labels_arr = np.array(labels);
        labels_arr.dump(result_files_prefix +'1'+ '_individual_labels.npy')
        labels_arr.dump(result_files_prefix +'2'+'_individual_labels.npy')
        pickle.dump(files, open(result_files_prefix +'1'+ '_filename.pkl', 'wb'))
        pickle.dump(files, open(result_files_prefix +'2'+ '_filename.pkl', 'wb'))
        pred_prob_arr = np.array(pred);
        pred_prob_arr.dump(result_files_prefix +'2'+ '_pred_new.npy')
        pred_manual_thresh_arr = np.array(pred_manual_thesh);
        pred_manual_thresh_arr.dump(result_files_prefix + '1'+'_pred_new.npy')

        
def generate_result_files_for_old_model(in_dir, out_dir, model_prefix):

    files_prefix = os.path.join(in_dir,'**', '*.png');
    result_files_prefix = os.path.join(out_dir, model_prefix);
    files = glob.glob(files_prefix, recursive=True);
    labels = [];
    pred = [];
    pred_manual_thesh = [];
    for file in files:
        filename = os.path.split(file)[1];
        label_str = filename[-5];
        label = int(label_str);
        labels.append(label);
        filename_split = filename[0:-6].split('-');
        pred_manual_thresh_str = filename_split[-1];
        pred_prob_str = filename_split[-3];
        pred_manual_thesh.append(np.array([0, int(pred_manual_thresh_str)]));
        pred.append(np.array([0, float(pred_prob_str)]));
        print(file)
        print(label)
        print(float(pred_prob_str))
        print(int(pred_manual_thresh_str))

    labels_arr = np.array(labels);
    labels_arr.dump(result_files_prefix +'1'+ '_individual_labels.npy')
    labels_arr.dump(result_files_prefix +'2'+'_individual_labels.npy')
    pickle.dump(files, open(result_files_prefix +'1'+ '_filename.pkl', 'wb'))
    pickle.dump(files, open(result_files_prefix +'2'+ '_filename.pkl', 'wb'))
    pred_prob_arr = np.array(pred);
    pred_prob_arr.dump(result_files_prefix +'2'+ '_pred_new.npy')
    pred_manual_thresh_arr = np.array(pred_manual_thesh);
    pred_manual_thresh_arr.dump(result_files_prefix + '1'+'_pred_new.npy')

def generate_labels_files_all_types(dir, model_prefix):
    types = ['coad', 'brca', 'read', 'luad', 'uvm', 'lusc', 'ucec', 'stad', 'blca', 'paad', 'prad', 'cesc', 'skcm'];

    for ctype in types:
        print(ctype);

        result_files_prefix = os.path.join(dir, ctype, model_prefix);
        lbl = np.load(result_files_prefix + '_individual_labels.npy');        
        filenames_old = pickle.load(open(result_files_prefix + '_filename.pkl', "rb"));
        filenames_old = np.array(filenames_old);

        filenames_new = [];
        labels_new = [];

        for file_old in filenames_old:
            print(file_old);
            file_pattern = file_old[:-40]  + '*.png';
            file_new = glob.glob(file_pattern)[0];
            print(file_new);
            filenames_new.append(file_new);
            label_new_str = file_new[-5];
            label_new = int(label_new_str);
            labels_new.append(label_new);            

        labels_arr = np.array(labels_new);
        labels_arr.dump(result_files_prefix + '_individual_labels.npy')
        pickle.dump(filenames_new, open(result_files_prefix + '_filename.pkl', 'wb'))

if __name__ == "__main__":
    #sys.argv

    in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes";
    out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes";

    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes_clr_norm";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes_clr_norm";

    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_bytype_dk5e-1_cont2_epoch_0003" # auc =  0.8548
    #threshold = 0.15; # youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_e0"; # auc = 0.8494
    #threshold = 0.58; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_e2"; # auc =  0.8535
    #threshold = 0.3; # youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_semiauto_e81"; # auc =  0.7587
    #threshold = 0.52; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_semiauto_e23"; # auc =  0.78177
    #threshold = 0.5; # youden
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_manual_e22"; # 0.7837
    #threshold = 0.25; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_manual_e19"; # auc =  0.7921
    #threshold = 0.37; # youden
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_manual_e5"; # auc =  
    #threshold = 0.5;
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_manual_e4"; # auc =  
    #threshold = 0.5;

    ##model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_wd5e-4_d75_luad_manual_epoch39"; # auc =  0.7851

    ##model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_bytype_epoch_0001" # auc =  0.8498
    ##model_prefix = "tcga_incv4_val-strat_all_bytype_inner-dk85e-2_cont4_epoch_0047" # auc =  0.840 
    ##model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_bytype_dk5-1_cont2_e4"; # auc =  0.4781
    ##model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_bytype2_e0" # auc = 0.5660
    ##model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_bytype_wfixed-types_full_e0" # auc =  0.5004

    ##model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_bytype_e1" # 
    ##model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_bytype_inner-dk85e-2_cont4_e9"
    ##model_prefix = "tcga_vgg16_bytype_tmp"
    #threshold = 0.2
    ######################################################################################

    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/val_luad_strat";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/val_luad_strat";
    #dataset_name = "val_luad_strat";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/val_skcm";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/val_skcm";
    #dataset_name = "val_skcm";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/val_luad";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/val_luad";
    #dataset_name = "val_luad";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/ucec_batch_1";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/ucec_batch_1";
    #dataset_name = "ucec_batch_1";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/paad_batch_1n2";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/paad_batch_1n2";
    #dataset_name = "paad_batch_1n2";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/coad_batch_1";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/coad_batch_1";
    #dataset_name = "coad_batch_1";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes/luad_additional+";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes/luad_additional+";
    #dataset_name = "luad_additional+";
    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes/brca_additional+";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes/brca_additional+";
    #dataset_name = "brca_additional+";

    #threshold = 0.5;

    ##model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_bytype_dk5e-1_cont2_e3" # cutoff =  0.15
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_bytype_dk5e-1_cont2_epoch_0003" # cutoff =  0.15
    ##threshold = 0.5;
    #threshold = 0.2;
    #threshold = 0.15; # youden
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
    #model_prefix = "tcga_incv4_crop100_noBN_d75_val-strat_all_tune-manual2_e1"; # cutoff_youden = 0.43216982 cutoff_distance = 0.43216982  auc = 0.95644 
    #threshold = 0.5;
    #threshold = 0.43;# youden
    #model_prefix = "tcga_vgg16_crop100_val-strat_all_tune-manual_e6"; # cutoff_youden = 0.0710279 cutoff_distance = 0.0710279 auc = 0.9530
    ##threshold = 0.5;
    #threshold = 0.07;# youden

    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_semi_e0"; # cutoff_youden = 0.4943972 cutoff_distance = 0.5410269 auc = 0.9427495291902072
    ##threshold = 0.5;
    #threshold = 0.49; # youden
    ##threshold = 0.54; # distance
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_semi_e0"; # cutoff_youden = 0.6106335 cutoff_distance = 0.6106335 auc = 0.9433193297600078
    ##threshold = 0.5;
    #threshold = 0.61;# youden

    ########### new ##############
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

    ### ok
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


    #model_prefix = "baseline1"; 
    #threshold = 0.5;

    #model_prefix = "baseline2"; # cutoff_youden = 0.26 
    #threshold = 0.26;# youden

    #model_prefix = "menndel"; #youden = 0.258 = cutoff_distance 
    #threshold = 0.5;
    ##threshold = 0.26; # youden

    #model_prefix = "menndel_v2"; # cutoff_youden =  0.1794 cutoff_distance =  0.1794
    ##threshold = 0.5;
    #threshold = 0.18; # youden

    model_prefix = "vgg-context"; # cutoff_youden =  0.44101143 cutoff_distance =  0.44101143
    #threshold = 0.5;
    threshold = 0.44; # youden

    ###print(threshold )
    #process_results(in_dir, out_dir, model_prefix, dataset_name, threshold);
    process_results_separate_types(in_dir, out_dir, model_prefix, threshold);
    ##process_results_separate_types_sm(in_dir, out_dir, model_prefix, threshold);
    #process_results_separate_types_auc(in_dir, out_dir, model_prefix);

    #threshold = 0.85; # youden

    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes";
    ##out_dir = "/pylon5/ac3uump/shahira/tcga/misclassified/all_ctypes"
    #out_dir = "/pylon5/ac3uump/shahira/tcga/misclassified/all_ctypes_vgg_all"

    #in_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes_clr_norm";
    #out_dir = "/pylon5/ac3uump/shahira/tcga/misclassified/all_ctypes_clr_norm_vgg_all"


    #copy_patches_wpred_separate_types(in_dir, out_dir, model_prefix, threshold);

    #model_prefix_list = [];
    #model_prefix_list.append("baseline2");
    #model_prefix_list.append("tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_manual_e5");
    #model_prefix_list.append("tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_manual_e4");
    #model_prefix_list.append("tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_semi_e0");
    #model_prefix_list.append("tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_semi_e0");
    #model_prefix_list.append("tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_e2");
    #model_prefix_list.append("tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_e0");
    #label_list = [];
    #label_list.append("Baseline");
    #label_list.append("Model-A1");
    #label_list.append("Model-A2");
    #label_list.append("Model-B1");
    #label_list.append("Model-B2");
    #label_list.append("Model-C1");
    #label_list.append("Model-C2");
    #process_results_separate_types_roc_curve(in_dir, model_prefix_list, label_list);

    #in_dir = '/pylon5/ac3uump/lhou/patches_val_allcancertype_luad_stratified_corrected'
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes";
    #in_dir = '/pylon5/ac3uump/shahira/tcga/datasets/patches_val_allcancertype_luad_stratified_corrected'
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes";
    #model_prefix = 'baseline'
    #generate_result_files_all_types_for_old_model(in_dir, out_dir, model_prefix);

    #in_dir = '/pylon5/ac3uump/shahira/tcga/datasets/patches_val_allcancertype_luad_stratified_corrected/luad_additional+'
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes/luad_additional+";
    #in_dir = '/pylon5/ac3uump/shahira/tcga/datasets/patches_val_allcancertype_luad_stratified_corrected/brca_additional+'
    #out_dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes/brca_additional+";
    #model_prefix = 'baseline'
    #generate_result_files_for_old_model(in_dir, out_dir, model_prefix)

    #dir = "/pylon5/ac3uump/shahira/tcga/test_out/all_ctypes";
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_bytype_dk5e-1_cont2_epoch_0003"
    ###model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_manual_e5";
    ###model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_manual_e4";
    ###model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_semi_e0";
    ###model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_semi_e0";
    ###model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_all_e2";
    ###model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_all_e0";
    ###model_prefix = 'baseline1';
    ##model_prefix = 'baseline2';
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_semiauto_e81"; # cutoff =  0.52  
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_semiauto_e23"; # cutoff =  0.50
    #model_prefix = "tcga_incv4_2class_adam_b128_CEloss_lr5e-5_crop100_noBN_d75_val-strat_luad_manual_e22"; # cutoff =  0.25
    #model_prefix = "tcga_vgg16_2class_adam_b128_CEloss_lr5e-5_crop100_val-strat_luad_manual_e19";  # cutoff =  0.37

    #generate_labels_files_all_types(dir, model_prefix)
