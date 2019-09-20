import numpy as np
import scipy.ndimage as ndimage

def get_tissue_map(wht):
    wht = ndimage.gaussian_filter(wht, sigma=(2,2), order=0, mode='reflect')
    return (wht > 12).astype(np.uint8)
