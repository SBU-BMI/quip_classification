import numpy as np
import h5py
import openslide
import sys
import os
import random
from PIL import Image

coord_txt = sys.argv[1];
output_folder = './patches/' + sys.argv[1].split('/')[-1];
svs_full_path = sys.argv[2];
patch_size_20X = 500;
max_n_patch = 2000;

try:
    mpp_w_h = os.popen('bash ../util/get_mpp_w_h.sh {}'.format(svs_full_path)).read();
    if len(mpp_w_h.split()) != 3:
        print '{}: mpp_w_h wrong'.format(svs_full_path);
        exit(1);

    mpp = float(mpp_w_h.split()[0]);
    width = int(mpp_w_h.split()[1]);
    height = int(mpp_w_h.split()[2]);
    if (mpp < 0.01 or width < 1 or height < 1):
        print '{}: mpp, width, height wrong'.format(svs_full_path);
        exit(1);
except:
    print '{}: exception caught'.format(svs_full_path);
    exit(1);

mag = 10.0 / mpp;
pw = int(patch_size_20X * mag / 20);

if not os.path.exists(output_folder):
    os.mkdir(output_folder);
fid = open(output_folder + '/label.txt', 'w');

obj_ids = 0;
lines = [line.rstrip('\n') for line in open(coord_txt)];
if len(lines) > max_n_patch:
    lines = random.sample(lines, max_n_patch)

for _, line in enumerate(lines):
    fields = line.split('\t');
    iid = fields[0];
    calc_width = int(fields[6]);
    calc_height = int(fields[7]);
    tot_width = int(fields[8]);
    tot_height = int(fields[9]);
    x = int(float(fields[2]) * calc_width);
    y = int(float(fields[3]) * calc_height);
    pred = float(fields[4]);
    label = int(fields[5]);
    fname = output_folder + '/{}.png'.format(obj_ids);
    print '{}, {}, {}, {}, {}, {}'.format(svs_full_path, (x-pw/2), (y-pw/2), pw, pw, fname)

    os.system('bash ../util/save_tile.sh {} {} {} {} {} {}'.format(svs_full_path, (x-pw/2), (y-pw/2), pw, pw, fname));
    patch = Image.open(fname).resize((patch_size_20X, patch_size_20X), Image.ANTIALIAS).convert('RGB');
    patch.save(fname);

    fid.write('{}.png {} {} {} {} {:.3f}\n'.format(obj_ids, label, iid, x, y, pred));
    fid.flush();

    obj_ids += 1;

fid.close();
