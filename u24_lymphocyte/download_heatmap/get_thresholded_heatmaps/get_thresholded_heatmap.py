import numpy as np
from get_labeled_im import *
from get_whiteness_im import *
from get_tissue_map import *
import sys
from scipy import misc

svs_name = sys.argv[1]
width = int(sys.argv[2])
height = int(sys.argv[3])
pred_file = sys.argv[4]
color_file = sys.argv[5]
output_path = sys.argv[6]

pred, necr, patch_size = get_labeled_im(pred_file)
whiteness, blackness, redness = get_whiteness_im(color_file)

im = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
im[:, :, 0] = 255 * (pred > 0.5).astype(np.uint8) * (blackness>30).astype(np.uint8) * (redness<0.15).astype(np.uint8)
im[:, :, 1] = 255 * necr
im[:, :, 2] = 255 * get_tissue_map(whiteness) * (im[:, :, 0] == 0).astype(np.uint8)

im = np.swapaxes(im, 0, 1)
misc.imsave('{}/{}.png'.format(output_path, svs_name), im)
