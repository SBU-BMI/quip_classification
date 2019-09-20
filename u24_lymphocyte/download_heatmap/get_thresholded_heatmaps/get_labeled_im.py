import os
import sys
import numpy as np
from scipy import signal, misc
from scipy.cluster.hierarchy import linkage
from skimage import draw

def get_labeled_im(pred_f):
    pred_data = np.loadtxt(pred_f).astype(np.float32)
    x = pred_data[:, 0]
    y = pred_data[:, 1]
    l = pred_data[:, 2]
    n = pred_data[:, 3]

    calc_width = x.min() + x.max()
    calc_height = y.min() + y.max()
    patch_size = (x.min() + x.max()) / len(np.unique(x))

    x = np.round((x + patch_size/2.0) / patch_size)
    y = np.round((y + patch_size/2.0) / patch_size)

    iml = np.zeros((int(x.max()), int(y.max())), dtype=np.float32)
    imn = np.zeros((int(x.max()), int(y.max())), dtype=np.float32)
    for iter in range(len(x)):
        iml[int(x[iter]-1), int(y[iter]-1)] = l[iter]
        imn[int(x[iter]-1), int(y[iter]-1)] = n[iter]

    return iml, imn, patch_size
