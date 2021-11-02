import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy.optimize as opt
from sampling import sample_oscillator_n

def mksamps(nsamps, nx):
    xs = sample_oscillator_n(nsamps)
    return xs

def klber(p,q):
    return p * (np.log(p) - np.log(q)) + (1-p)*(np.log(1-p) - np.log(1-q))


def klber_ub(p, eps):
    """
    Computes the upper bound on q in the implicit inequality
    $KL_{ber}(p||q) \le \epsilon$, when $p$ and $\epsilon$ are fixed.
    """
    return opt.brentq(lambda q: klber(p,q) - eps, p, 1-1e-6)


def kl_centered_normal(cov0, cov1):
    """
    computes the KL divergence KL(N(0,cov0)||N(0,cov1)).
    """
    n = np.shape(cov0)[0]
    print('log det 0')
    s, ld0 = np.linalg.slogdet(cov0)
    print('log det 1')
    s, ld1 = np.linalg.slogdet(cov1)
    print('inv')
    cov1_inv = np.linalg.inv(cov1)
    print('trace')
    tr = np.trace(np.matmul(cov1_inv, cov0))
    print('trace of k matrix: {:.5g}'.format(np.trace(cov1)))
    kl = 0.5 * (ld1 - ld0 + tr - n)
    return kl

def kl_cfun(xs, kfun, sig0=-1):
    ns, nx = np.shape(xs)
    kmat = kfun(xs, xs)
    if sig0 == -1:
        kmat_inv = np.linalg.inv(kmat+1e-9*np.eye(ns))
        tr = np.trace(kmat_inv)
        print(tr)
        sig0 = ns / tr
        print('computed noise level: {:.5g}'.format(sig0))
    noisemat = sig0 * np.eye(ns)
    prior_cov = kmat + noisemat
    L = np.linalg.cholesky(prior_cov)
    eigvals = np.diag(L)**2
    ld = np.real(np.sum(np.log(eigvals))) # log det
    tr = np.trace(np.linalg.inv(prior_cov)) # trace of inverse
    lsig0 = np.log(sig0)
    #import pdb; pdb.set_trace()
    return 0.5*(ld - ns*lsig0 + sig0*tr - ns)

def x2cov(xs, kfun):
    nx = np.shape(xs)[0]
    cmat = np.zeros((nx,nx))
    for (i1, x1) in enumerate(xs):
        for (i2, x2) in enumerate(xs):
            cmat[i1,i2] = kfun(x1,x2)
    return cmat

def make_posterior_kernel(kfun, xs, sig0):
    ns = np.shape(xs)[0]
    kmat = kfun(xs, xs) + sig0*np.eye(ns)
    kmat_inv = np.linalg.inv(kmat)
    def posterior_kernel(z1, z2):
        kvec1 = kfun(xs, z1)
        #kvec1.reshape((ns,1))
        kvec2 = kfun(xs, z2)
        #kvec2.reshape((ns,1))
        # kvec1 = np.zeros((ns,1))
        # kvec2 = np.zeros((ns,1))
        # for (i,x) in enumerate(xs):
        #     kvec1[i,0] = kfun(z1, x)
        #     kvec2[i,0] = kfun(z2, x)
        kv = np.matmul(kvec1.T, np.matmul(kmat_inv, kvec2))
        return kfun(z1, z2) - kv
    return posterior_kernel


def kfun_nonbroadcastable(z1,z2):
    return np.exp(-np.linalg.norm(z1-z2)**2/2)
    #return np.exp(np.dot(z1,z2))
    #return np.power(np.dot(z1,z2)+1, 3)


def kfun_se(xs, ys, sig=1/5):
    if len(np.shape(xs)) == 1:
        nxs = np.shape(xs)[0]
        xs = xs.reshape(1,nxs)
    if len(np.shape(ys)) == 1:
        nys = np.shape(ys)[0]
        ys = ys.reshape(1,nys)
    return np.exp(-np.linalg.norm(xs[:,np.newaxis]-ys, axis=-1)**2/(2*sig**2))

def kfun_exp(xs, ys, sig=1/3):
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

kfun = kfun_se

nsamps=100
nx=2


eta = .25 # for SE kernel
sig0=eta/chi2.isf(.001,df=1)
stochastic_err = chi2.sf(eta/sig0, df=1)
print('Threshold: {:.3g}'.format(eta))
print('stochastic err: {:.3g}'.format(stochastic_err))
print('Noise level: {:.3g}'.format(sig0))

print('Computing samples')
xs_raw = mksamps(nsamps,2)
# normalize
xs_mean = np.mean(xs_raw, axis=0)
xs_std = np.std(xs_raw, axis=0)
xs = (xs_raw - xs_mean) / xs_std

# # Nystrom approximation (not yet implemented)
# nsamps_nys = 50
# 
# kmm = kfun(xs[:nsamps_nys, :], xs[:nsamps_nys, :])
# kmn = kfun(xs[:nsamps_nys, :], xs[nsamps_nys:, :])
# import pdb; pdb.set_trace()
# priorcov_nys = np.matmul(kmn, np.linalg.solve(kmm+sig0*np.eye(nsamps_nys), kmn))

print('building kernel matrix')
noisemat = sig0*np.eye(nsamps)
kmat = kfun(xs, xs)
priorcov = kmat + noisemat
print('kernel matrix complete')

postcov = noisemat

print('Computing KL divergence')
kl = kl_centered_normal(postcov, priorcov)
print('KL divergence: {:.5g}'.format(kl))

delta = 1e-9

pbrhs = (kl + np.log((nsamps + 1)/delta)) / nsamps

print('RHS of PAC-Bayes bound: {:.5g}'.format(pbrhs))

errbnd_stochastic = klber_ub(stochastic_err, pbrhs)
errbnd_mean = errbnd_stochastic / chi2.cdf(1,1)

print('PAC-Bayes upper bound: {:.3g}'.format(errbnd_mean))


#kfun_post = make_posterior_kernel(kfun, xs, sig0)

#import pdb; pdb.set_trace()

print('Plotting data and contour')
ngrid=100
gridwidth=3
xgrid = np.linspace(-gridwidth,gridwidth,ngrid)
ygrid = np.linspace(-gridwidth,gridwidth,ngrid)
X,Y = np.meshgrid(xgrid,ygrid)

#gridpoints = np.vstack((X.flat,Y.flat)).T

Z = np.zeros(np.shape(X))
kmat_inv = np.linalg.inv(priorcov)

for (i, xi) in enumerate(xgrid):
    for (j, yj) in enumerate(ygrid):
        v = np.array([xi, yj])
        kvec = kfun(xs,v)
        kv = np.matmul(kvec.T,np.matmul(kmat_inv,kvec))
        zval = kfun(v,v)[0,0] - kv[0,0]
        #import pdb; pdb.set_trace()
        Z[i,j] = zval

plt.plot(xs[:,1],xs[:,0],linestyle='none', marker='.')
plt.contour(X,Y,Z, levels=[eta], colors='green')
plt.show()

#import pdb; pdb.set_trace()
