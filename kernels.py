import numpy as np

def kfun_se(xs, ys, sig=1/4):
    if len(np.shape(xs)) == 1:
        nxs = np.shape(xs)[0]
        xs = xs.reshape(1,nxs)
    if len(np.shape(ys)) == 1:
        nys = np.shape(ys)[0]
        ys = ys.reshape(1,nys)
    return np.exp(-np.linalg.norm(xs[:,np.newaxis]-ys, axis=-1)**2/(2*sig**2))

def kfun_se_col(xs, i, sig=1/4):
    if len(np.shape(xs)) == 1:
        nxs = np.shape(xs)[0]
        xs = xs.reshape(1,nxs)
    if len(np.shape(ys)) == 1:
        nys = np.shape(ys)[0]
        ys = ys.reshape(1,nys)
    return np.exp(-np.linalg.norm(xs-xs[:,i], axis=-1)**2/(2*sig**2))

def kfun_exp(xs, ys, sig=1/2):
    if len(np.shape(xs)) == 1:
        nxs = np.shape(xs)[0]
        xs = xs.reshape(1,nxs)
    if len(np.shape(ys)) == 1:
        nys = np.shape(ys)[0]
        ys = ys.reshape(1,nys)
    return np.exp(np.dot(xs, ys.T)/sig)


def kfun_poly(xs, ys):
    if len(np.shape(xs)) == 1:
        nxs = np.shape(xs)[0]
        xs = xs.reshape(1,nxs)
    if len(np.shape(ys)) == 1:
        nys = np.shape(ys)[0]
        ys = ys.reshape(1,nys)
    return np.power(np.dot(xs, ys.T)+1, 10)
