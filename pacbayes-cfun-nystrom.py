import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy.optimize as opt
from sampling import sample_oscillator_n
import scipy as sp
import time


from pacbayes_utils import (
        mksamps, 
        klber, 
        klber_ub, 
        kl_centered_normal,
        kfun_se, 
        kfun_exp,
        kfun_poly
        )


kfun = kfun_se

nsamps=1000
nx=2


eta = .05
sig0=eta/chi2.isf(.001,df=1)
stochastic_err = chi2.sf(eta/sig0, df=1)
print('Threshold: {:.3g}'.format(eta))
print('stochastic err: {:.3g}'.format(stochastic_err))
print('Noise level: {:.3g}'.format(sig0))

print('Computing samples')
tic_sampling = time.perf_counter()
xs_raw = mksamps(nsamps,2)
toc_sampling = time.perf_counter()
# normalize
xs_mean = np.mean(xs_raw, axis=0)
xs_std = np.std(xs_raw, axis=0)
xs = (xs_raw - xs_mean) / xs_std

# Nystrom approximation
nsamps_nys = 100

print('Building Nystrom approximation')
tic_nys = time.perf_counter()
kmat = kfun(xs,xs)
print(np.shape(kmat))
kmm = kfun(xs[:nsamps_nys, :], xs[:nsamps_nys, :]) + sig0*np.eye(nsamps_nys)
kmn = kfun(xs[:nsamps_nys, :], xs)
kmnnm = np.matmul(kmn, kmn.T)
toc_nys = time.perf_counter()
e0 = sp.linalg.eigvalsh(kmat)
e1 = sp.linalg.eigvalsh(kmm)*nsamps/nsamps_nys
e2 = sp.linalg.eigvalsh(np.matmul(kmn,kmn.T), kmm)
plt.plot(np.flip(e0),label='true evals')
plt.plot(np.flip(e1),label='scaled kmm')
plt.plot(np.flip(e2),label='generalized evp')
plt.legend()
plt.show()
print('Nystrom approximation complete')

#priorcov_nys = np.matmul(kmn.T, np.linalg.solve(kmm, kmn)) + sig0*np.eye(nsamps)
#nys_eval_matrix = np.matmul(np.linalg.solve(kmm, kmn), kmn.T) + sig0*np.eye(nsamps_nys)
# evals_full = np.real(np.linalg.eigvalsh(priorcov))
# evals_kmm = np.real(np.linalg.eigvalsh(kmm)) * nsamps / nsamps_nys
# evals_nys = np.real(np.linalg.eigvalsh(priorcov_nys))
# #evals_nys_small = np.flip(np.real(np.linalg.eigvals(nys_eval_matrix)))
# plt.plot(np.flip(evals_full), label='$K_{nn}$')
# plt.plot(np.flip(evals_nys), label='$K_{nm}K_{mm}^{-1}K_{mn}$')
# plt.plot(np.flip(evals_nys_small), label='$K_{mn}K_{nm}K_{mm}^{-1}$')
# plt.legend()
# plt.title('Eigenvalues of Matrices Associated with the Nystrom Approximation')
# plt.show()



print('Computing KL divergence')
tic_kl = time.perf_counter()
evals_nys = sp.linalg.eigvalsh(np.matmul(kmn,kmn.T), kmm)
ld_nys = np.sum(np.log(evals_nys + sig0)) + (nsamps - nsamps_nys)*np.log(sig0)
invtrace_nys = np.sum(np.power(evals_nys + sig0, -1)) + (nsamps - nsamps_nys) / sig0
kl = 0.5 * (ld_nys - nsamps * np.log(sig0) + sig0*invtrace_nys - nsamps)
toc_kl = time.perf_counter()
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
tic_plot = time.perf_counter()
ngrid=100
gridwidth=3
xgrid = np.linspace(-gridwidth,gridwidth,ngrid)
ygrid = np.linspace(-gridwidth,gridwidth,ngrid)
X,Y = np.meshgrid(xgrid,ygrid)

Z = np.zeros(np.shape(X))

kinv = np.linalg.inv(sig0*kmm + kmnnm)
kchol = np.linalg.cholesky(sig0*kmm + kmnnm)
for (i, xi) in enumerate(xgrid):
    for (j, yj) in enumerate(ygrid):
        v = np.array([xi, yj])
        kxx = kfun(v,v)
        knx = kfun(xs,v)
        kxnnx = np.matmul(knx.T, knx)
        knxnm = np.matmul(knx.T, kmn.T)
        #kv = sig0**(-1)*np.matmul(knxnm, np.matmul(kinv, knxnm.T))
        kv1 = sp.linalg.solve_triangular(kchol, knxnm.T)
        kv2 = sp.linalg.solve_triangular(kchol.T, kv1)
        kv = sig0**(-1) * kv2
        zval = kxx - sig0**(-1)*kxnnx + kv
        Z[i,j] = zval[0,0]

plt.plot(xs[:,1],xs[:,0],linestyle='none', marker='.')
plt.contour(X,Y,Z, levels=[eta], colors='green')
plt.title('$N={:d}$, $N_{{nys}}={:d}$, $\epsilon={:.3g}$, $\delta={:.3g}$'.format(nsamps, nsamps_nys, errbnd_mean, delta))
toc_plot = time.perf_counter()
plt.savefig('kernel_cfun_nystrom_{:d}_nys{:d}.png'.format(nsamps, nsamps_nys))

print('--- timing ---')
print('sampling: {:.4f} s'.format(toc_sampling - tic_sampling))
print('nystrom : {:.4f} s'.format(toc_nys - tic_nys))
print('KL div  : {:.4f} s'.format(toc_kl - tic_kl))
print('plotting: {:.4f} s'.format(toc_plot - tic_plot))

#import pdb; pdb.set_trace()
