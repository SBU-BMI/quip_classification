import numpy as np
import openslide
import sys
import os
import random
from PIL import Image

def extract_rand_patch(svs_path, pat_xys):
    pw_20X = 100
    oslide = openslide.OpenSlide(svs_path)
    if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
        mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X])
    elif "XResolution" in oslide.properties:
        mag = 10.0 / float(oslide.properties["XResolution"])
    else:
        mag = 10.0 / float(0.254)

    pw = float(int(10 * pw_20X * mag / 20)) / 10.0
    pw3 = pw * 3
    pw3_20X = pw_20X * 3
    width = oslide.dimensions[0]
    height = oslide.dimensions[1]

    for pat_x, pat_y in pat_xys:
        x = 1 + pw * pat_x - pw
        y = 1 + pw * pat_y - pw
        if x <= 0 or y <= 0 or x + pw3 >= width or y + pw3 >= height:
            print('Patch extracting out of range {}/{} {}/{} {}/{}'.format(x, y, pat_x, pat_y, width, height))
            continue
        patch = oslide.read_region((x, y), 0, (pw3, pw3))
        patch = patch.resize((pw3_20X, pw3_20X), Image.ANTIALIAS)
        yield patch, x, y, pat_x, pat_y, pw3

til_dir = '/pylon5/ac3uump/lhou/til-maps/TIL_maps_after_thres_v1/'
wsi_dir = '/pylon5/ac3uump/lhou/wsi/'
output_root = '/pylon5/ac3uump/lhou/patches/'

til_list = []
for root, dirnames, filenames in os.walk(til_dir):
    for filename in filenames:
        if filename.endswith('.png'):
            til_list.append(os.path.join(root, filename))
wsi_list = []
for root, dirnames, filenames in os.walk(wsi_dir):
    for filename in filenames:
        if filename.endswith('.svs'):
            wsi_list.append(os.path.join(root, filename))

til_dict = dict(zip(['.'.join([x.split('/')[-2], x.split('/')[-1].split('.')[0]]) for x in til_list], til_list))
wsi_dict = dict(zip(['.'.join([x.split('/')[-2], x.split('/')[-1].split('.')[0]]) for x in wsi_list], wsi_list))

keys2remove = []
for key, value in til_dict.items():
    if key not in wsi_dict:
        keys2remove.append(key)
for key in keys2remove:
    del til_dict[key]

keys2remove = []
for key, value in wsi_dict.items():
    if key not in til_dict:
        keys2remove.append(key)
for key in keys2remove:
    del wsi_dict[key]

print('Number of available WSIs: {}'.format(len(wsi_dict)))

wsi_dict_items = list(wsi_dict.items())
random.shuffle(wsi_dict_items)
for key, value in wsi_dict_items:
    svs_path = value
    til_path = til_dict[key]
    ctype = os.path.basename(os.path.dirname(svs_path))
    svs_name = os.path.basename(til_path)[:-len('.png')]

    til_im = np.array(Image.open(til_path)).astype(np.uint8)
    pat_ys, pat_xs = np.where((til_im > 1).sum(axis=-1) > 0)
    pat_xys = random.sample(list(zip(pat_xs, pat_ys)), 20)

    output_dir = os.path.join(output_root, ctype, svs_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        for patch, x, y, pat_x, pat_y, pw3 in extract_rand_patch(svs_path, pat_xys):
            label = int(til_im[pat_y, pat_x, 0] > 1)
            save_name = '{}_{}_{}_{}_{}_{}_{}.png'.format(svs_name, x, y, pat_x, pat_y, pw3, label)
            patch.save(os.path.join(output_dir, save_name))
    except:
        print('Error in {}: exception caught'.format(svs_path))

