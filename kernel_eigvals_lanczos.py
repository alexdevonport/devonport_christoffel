import numpy as np
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
from kernels import *
import logging

def make_kmat_op(kfun, xs, nx):
    assert np.shape(xs)[1] == nx, "data dimension mismatch"
    ns = np.shape(xs)[0]
    def kmat_op(v): 
        r = np.zeros(ns)
        for (i,xi) in enumerate(xs):
            r[i] = np.dot(v, kfun(xs, xi))
        return r
    op = LinearOperator((ns,ns), matvec=kmat_op)
    return op

def kernel_eigvals_lanczos(kfun, xs, nx, evals_frac=0.25, maxevals=1e8):
    ns = np.shape(xs)[0]
    nevals = int(min(ns * evals_frac, maxevals))
    logging.info('evals to compute: {:d}'.format(nevals))
    # kop = make_kmat_op(kfun, xs, nx)
    # eigs_op = scipy.sparse.linalg.eigsh(kop, return_eigenvectors=False,k=nevals)
    kmat = kfun(xs, xs)
    eigs_op = scipy.sparse.linalg.eigsh(kmat, return_eigenvectors=False,k=nevals)
    eigs_overapprox = np.ones(ns) * eigs_op[0]
    eigs_overapprox[:nevals] = np.flip(eigs_op)
    return eigs_overapprox

