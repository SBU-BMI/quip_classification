import glob
import numpy as np
import os
from create_dummy_heatmap_for_wsi import * 

svs_dir = '../../wsi' 
#images_csv_path = './images.csv'
images_csv_filepath = './images.csv'
out_dir = '../../dummy_heatmaps_output'
heatmap_txt_dir = os.path.join(out_dir, 'heatmap_txt')
heatmap_json_dir = os.path.join(out_dir, 'heatmap_json')
failed_log_filepath = os.path.join(out_dir, 'failed_log.txt')

if(not os.path.exists(out_dir)):
    os.mkdir(out_dir)
if(not os.path.exists(heatmap_txt_dir)):
    os.mkdir(heatmap_txt_dir)
if(not os.path.exists(heatmap_json_dir)):
    os.mkdir(heatmap_json_dir)


with open(failed_log_filepath, 'w+') as failed_log:
    failed_log.write('subject_id,case_id,file-location\n')
    failed_log.flush()
    svs_list=np.loadtxt(images_csv_filepath,delimiter=',',dtype=np.str)
    for i in range(1, svs_list.shape[0]):
        subject_id = svs_list[i][0]
        case_id = svs_list[i][1]
        ref_svs_filepath = svs_list[i][2]
        real_svs_filepath = glob.glob(os.path.join(svs_dir, '**', case_id+'*.svs'))
        if(len(real_svs_filepath)==0):
            failed_log.write(subject_id + ','+case_id+','+ref_svs_filepath+'\n')
            failed_log.flush()
            print('** ref_svs_filepath', ref_svs_filepath, ' not found')
            continue
        real_svs_filepath = real_svs_filepath[0]
        print('real_svs_filepath', real_svs_filepath)        
        heatmap_txt_filepath = generate_dummy_heatmap_txt(real_svs_filepath, heatmap_txt_dir)         
        generate_dummy_json(real_svs_filepath, heatmap_json_dir, heatmap_txt_filepath, case_id, subject_id)
        #break

               




