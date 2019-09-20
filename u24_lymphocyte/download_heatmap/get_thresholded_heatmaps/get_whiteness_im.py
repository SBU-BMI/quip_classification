import numpy as np

def get_whiteness_im(white_file):
    data = np.loadtxt(white_file).astype(np.float32)
    x = data[:, 0]
    y = data[:, 1]
    w = data[:, 2]
    b = data[:, 3]
    r = data[:, 4]

    step = (x.min() + x.max()) / len(np.unique(x))

    x = np.round((x + step/2.0) / step)
    y = np.round((y + step/2.0) / step)

    whiteness = np.zeros((int(x.max()), int(y.max())), dtype=np.uint8)
    blackness = np.zeros((int(x.max()), int(y.max())), dtype=np.uint8)
    redness = np.zeros((int(x.max()), int(y.max())), dtype=np.uint8)

    for iter in range(len(x)):
        whiteness[int(x[iter]-1), int(y[iter]-1)] = w[iter]
        blackness[int(x[iter]-1), int(y[iter]-1)] = b[iter]
        redness[int(x[iter]-1), int(y[iter]-1)] = r[iter]

    return whiteness, blackness, redness
