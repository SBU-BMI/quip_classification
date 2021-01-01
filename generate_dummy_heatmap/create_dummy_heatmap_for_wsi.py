import numpy as np
import skimage.io as io
import os
import glob
import openslide
import sys
from PIL import Image
import datetime
import json
from bson import json_util
import random

# heatmap txt settings
patch_size_40X = 200;

# heatmap json settings
is_shifted = False;
heatmap_name = 'grid_TIL-high_res';
n_heat = 2
cancer_type = 'quip';
heat_list=['lym', 'necrosis']
weight_list=['0.5', '0.5']

def generate_dummy_heatmap_txt(svs_path, out_dir):

    file = svs_path
    slide_name = os.path.splitext(os.path.basename(file))[0]
    print(slide_name)
    oslide = openslide.OpenSlide(file);

    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
       mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
    elif "XResolution" in oslide.properties:
       mag = 10.0 / float(oslide.properties["XResolution"]);
    elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
       mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
    else:
       mag = 10.0 / float(0.254);
    pw = int(patch_size_40X * mag / 40);
    width = oslide.dimensions[0];
    height = oslide.dimensions[1];
    heatmap_txt_filepath = os.path.join(out_dir, 'prediction-'+slide_name )
    with open(heatmap_txt_filepath , 'w+') as heatmap_txt:
        x_val = 1.0
        for x in range(1, width, pw):
            x_val = 1.0-x_val
            y_val = x_val
            for y in range(1, height, pw):
                if x + pw > width:
                    continue;
                if y + pw > height:
                    continue;
                heatmap_txt.write('{} {} {} 0\n'.format(x + pw//2, y + pw//2, y_val))
                y_val = 1.0-y_val

    return heatmap_txt_filepath 

def generate_dummy_json(svs_path, out_dir, heatmap_txt_filepath, caseid, subjectid):
    pred_file_path = heatmap_txt_filepath;       
    pred_folder, pred_file_name = os.path.split(pred_file_path);
    filename = pred_file_name;
    imgfilename=svs_path;

    heatmapfile = os.path.join(out_dir, 'heatmap_' + filename.split('prediction-')[1] + '.json');
    metafile = os.path.join(out_dir, 'meta_' + filename.split('prediction-')[1] + '.json');

    oslide = openslide.OpenSlide(imgfilename);
    slide_width_openslide = oslide.dimensions[0];
    slide_height_openslide = oslide.dimensions[1];

    analysis_execution_id = 'highlym';
    x_arr = np.zeros(10000000, dtype=np.float64);
    y_arr = np.zeros(10000000, dtype=np.float64);
    score_arr = np.zeros(10000000, dtype=np.float64);
    score_set_arr = np.zeros(shape=(10000000, n_heat), dtype=np.float64);
    idx = 0;
    print('generating heatmap txt file')
    with open(pred_file_path) as infile:
        for line in infile:
            parts = line.split(' ');
            x = int(parts[0]);
            y = int(parts[1]);
            score = float(parts[2]);
            score_set = [float(i) for i in parts[2:2+n_heat]];

            x_arr[idx] = x;
            y_arr[idx] = y;
            score_arr[idx] = score;
            score_set_arr[idx] = score_set;
            idx += 1;

    x_arr = x_arr[:idx];
    y_arr = y_arr[:idx];
    score_arr = score_arr[:idx];
    score_set_arr = score_set_arr[:idx];

    patch_width = max(x_arr[1] - x_arr[0], y_arr[1] - y_arr[0]);
    patch_height = patch_width;

    slide_width = int(slide_width_openslide);
    slide_height = int(slide_height_openslide);

    x_arr = x_arr / slide_width;
    y_arr = y_arr / slide_height;
    patch_width = patch_width / slide_width;
    patch_height = patch_height / slide_height;
    patch_area = patch_width * slide_width * patch_height * slide_height;

    dict_img = {};
    dict_img['case_id'] = caseid;
    dict_img['subject_id'] = subjectid;

    dict_analysis = {};
    #dict_analysis['cancer_type'] = slide_type;
    dict_analysis['cancer_type'] = cancer_type
    dict_analysis['study_id'] = 'u24_tcga';
    dict_analysis['execution_id'] = heatmap_name;
    dict_analysis['source'] = 'computer';
    dict_analysis['computation'] = 'heatmap';


    if (is_shifted == True):
        shifted_x = -3*patch_width / 4.0;
        shifted_y = -3*patch_height / 4.0;
    else:
        shifted_x = 0;
        shifted_y = 0;

    print('generating json file')
    with open(heatmapfile, 'w') as f:
        for i_patch in range(idx):
            dict_patch = {};
            dict_patch['type'] = 'Feature';
            dict_patch['parent_id'] = 'self';
            #dict_patch['footprint'] = patch_area;
            dict_patch['footprint'] = 128000000 # need to be large to be visible at low resolution
            dict_patch['x'] = x_arr[i_patch] + shifted_x;
            dict_patch['y'] = y_arr[i_patch] + shifted_y;
            dict_patch['normalized'] = 'true';
            dict_patch['object_type'] = 'heatmap_multiple';

            x1 = dict_patch['x'] - patch_width/2;
            x2 = dict_patch['x'] + patch_width/2;
            y1 = dict_patch['y'] - patch_height/2;
            y2 = dict_patch['y'] + patch_height/2;
            dict_patch['bbox'] = [x1, y1, x2, y2];

            dict_geo = {};
            dict_geo['type'] = 'Polygon';
            dict_geo['coordinates'] = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]];
            dict_patch['geometry'] = dict_geo;

            dict_prop = {};
            dict_prop['metric_value'] = score_arr[i_patch];
            dict_prop['metric_type'] = 'tile_dice';
            dict_prop['human_mark'] = -1;

            dict_multiheat = {};
            dict_multiheat['human_weight'] = -1;
            dict_multiheat['weight_array'] = weight_list;
            dict_multiheat['heatname_array'] = heat_list;
            dict_multiheat['metric_array'] = score_set_arr[i_patch].tolist();

            dict_prop['multiheat_param'] = dict_multiheat;
            dict_patch['properties'] = dict_prop;

            dict_provenance = {};
            dict_provenance['image'] = dict_img;
            dict_provenance['analysis'] = dict_analysis;
            dict_patch['provenance'] = dict_provenance;

            dict_patch['date'] = datetime.datetime.now();

            json.dump(dict_patch, f, default=json_util.default);
            f.write('\n');

    print('generating meta file')
    with open(metafile, 'w') as mf:
        dict_meta = {};
        dict_meta['color'] = 'yellow';
        dict_meta['title'] = 'Heatmap-' + heatmap_name;
        dict_meta['image'] = dict_img;

        dict_meta_provenance = {};
        dict_meta_provenance['analysis_execution_id'] = heatmap_name;
        #dict_meta_provenance['cancer_type'] = slide_type;
        dict_meta_provenance['cancer_type'] = cancer_type;
        dict_meta_provenance['study_id'] = 'u24_tcga';
        dict_meta_provenance['type'] = 'computer';
        dict_meta['provenance'] = dict_meta_provenance;

        dict_meta['submit_date'] = datetime.datetime.now();
        dict_meta['randval'] = random.uniform(0,1);

        json.dump(dict_meta, mf, default=json_util.default);

