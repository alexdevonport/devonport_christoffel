import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy.special as sp
import scipy.optimize as opt
from sampling import sample_oscillator_n
from sklearn.preprocessing import PolynomialFeatures


from pacbayes_utils import (
        mksamps,
        klber,
        klber_ub,
        kl_centered_normal,
        )


nsamps=15000
nx=2
k = 10

nf = int(sp.binom(nx + k, nx))  # number of features (i.e. monomials)

targeteps = 0.05
eta = nf / targeteps # based on Markov inequality estimate for (1-eps) threshold

sig0=1e3

print('Nominal threshold: {:.3g}'.format(eta))
print('Prior variance: {:.3g}'.format(sig0))

print('Computing samples')
xs_raw = mksamps(nsamps,2)
# normalize
xs_mean = np.mean(xs_raw, axis=0)
xs_std = np.std(xs_raw, axis=0)
xs = (xs_raw - xs_mean) / (xs_std)

print('Constructing Christoffel Function')

pf = PolynomialFeatures(k)
zxs = pf.fit_transform(xs)

empirical_moment_matrix = sig0**(-1) * np.eye(nf) + np.matmul(zxs.T,zxs) / nsamps
inverse_empirical_moment_matrix = np.linalg.inv(empirical_moment_matrix)


print('Evaluating Christoffel function on data points')

cfun_evals = np.zeros((nsamps,))
for (i, zxi) in enumerate(zxs):
    cfun_evals[i] = np.matmul(zxi, np.matmul(inverse_empirical_moment_matrix, zxi.T))

print('max cfun data eval: {:.3g}'.format(np.max(cfun_evals)))

print('Evaluating empirical stochastic error')

point_errors = chi2.sf(eta / cfun_evals, 1)

stochastic_err = np.mean(point_errors)

print('stochastic error: {:.3g}'.format(stochastic_err))


print('Computing KL divergence')

prior_matrix = sig0 * np.eye(nf)

kl = kl_centered_normal(empirical_moment_matrix, prior_matrix)
print('KL divergence: {:.5g}'.format(kl))

delta = 1e-9

pbrhs = (kl + np.log((nsamps + 1)/delta)) / nsamps

print('RHS of PAC-Bayes bound: {:.5g}'.format(pbrhs))

errbnd_stochastic = klber_ub(stochastic_err, pbrhs)
errbnd_mean = errbnd_stochastic / chi2.cdf(1,1)
print('PAC-Bayes upper bound: {:.3g}'.format(errbnd_mean))


print('Plotting data and contour')
ngrid=100
gridwidth=3
xgrid = np.linspace(-gridwidth,gridwidth,ngrid)
ygrid = np.linspace(-gridwidth,gridwidth,ngrid)
X,Y = np.meshgrid(xgrid,ygrid)


Z = np.zeros(np.shape(X))

for (i, xi) in enumerate(xgrid):
    for (j, yj) in enumerate(ygrid):
        v = np.array([[xi, yj]])
        zv = pf.fit_transform(v)
        zval = np.matmul(zv, np.matmul(inverse_empirical_moment_matrix, zv.T))
        Z[i,j] = zval

plt.plot(xs[:,1],xs[:,0],linestyle='none', marker='.')
plt.contour(X,Y,Z, levels=[eta], colors='green')
plt.title('$N={:d}$, $\epsilon={:.3g}$, $\delta={:.3g}$, $\eta={:.5g}$'.format(nsamps, errbnd_mean, delta, eta))
plt.show()

