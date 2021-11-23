import numpy as np
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
from kernels import *
import logging
from multiprocessing import Pool, cpu_count

from sampling import sample_oscillator_n
from kernels import kfun_se

_func = None

def worker_init(func):
  global _func
  _func = func


def worker(x):
  return _func(x)


def xmap(func, iterable, processes=None):
  with Pool(processes, initializer=worker_init, initargs=(func,)) as p:
    return p.map(worker, iterable)


def make_kmat_op(kfun, xs, nx):
    assert np.shape(xs)[1] == nx, "data dimension mismatch"
    ns = np.shape(xs)[0]
    def kmat_op(v): 
        p = Pool(cpu_count())
        print(r'Using {} CPUs'.format(cpu_count()))
        return np.array(list(xmap(lambda i: np.dot(v, kfun(xs, xs[i])), np.arange(ns))))
        #return r
    op = LinearOperator((ns,ns), matvec=kmat_op)
    return op

def kernel_eigvals_lanczos(kfun, xs, nx, evals_frac=0.25, maxevals=1e8):
    ns = np.shape(xs)[0]
    nevals = int(min(ns * evals_frac, maxevals))
    logging.info('evals to compute: {:d}'.format(nevals))
    kop = make_kmat_op(kfun, xs, nx)
    eigs_op = scipy.sparse.linalg.eigsh(kop, return_eigenvectors=False,k=nevals)
    kmat = kfun(xs, xs)
    eigs_mat = scipy.sparse.linalg.eigsh(kmat, return_eigenvectors=False,k=nevals)
    print(eigs_op - eigs_mat)
    eigs_overapprox = np.ones(ns) * eigs_mat[0]
    eigs_overapprox[:nevals] = np.flip(eigs_mat)
    return eigs_overapprox

xs = sample_oscillator_n(100)

kernel_eigvals_lanczos(kfun_se, xs, 2)
