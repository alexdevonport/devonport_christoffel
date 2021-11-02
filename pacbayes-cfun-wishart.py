import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import math
from sampling import sample_oscillator_n
from sklearn.preprocessing import PolynomialFeatures

def mksamps(nsamps, nx):
    xs = sample_oscillator_n(nsamps)
    return np.matrix(xs)

def poly_map(xs, k):
    pf = PolynomialFeatures(k)
    poly_xs = pf.fit_transform(xs)
    return np.matrix(poly_xs)

def pacbayes_ellipse(xs, stochastic_err, delta, sig0, v1=None, n1=None):
    ns = np.shape(xs)[0]
    nx = np.shape(xs)[1]
    xmom = np.matmul(xs.T,xs)
    if v1 is None:
        v1 = sig0**(-1)*np.eye(nx) / nx
    if n1 is None:
        n1 = nx
    v0 = np.linalg.inv(np.linalg.inv(v1) + xmom)
    n0 = ns + nx
    posterior_mean = v0*n0
    eta = errlvl(xs, v0,n0, stochastic_err)
    #eta = nx  # something smart with chi squared quantile fn?
    #eta = chisq_quantile(1-1e-3,nx)
    stochastic_err = stochastic_error(xs, v0, n0, eta)
    print('eta: {:.3g}, serr = {:.3g}'.format(eta, stochastic_err))
    klpq = wishart_kl(v0, v1, n0, n1)
    pacbayes_rhs = (klpq + np.log((ns + 1) / delta)) / ns
    print(klpq)
    pacbayes_eps = klber_ub(stochastic_err, pacbayes_rhs)
    print(pacbayes_eps)
    return (posterior_mean, eta, pacbayes_eps)

def cfun_wis(x, pmean, k):
    s = np.shape(x)
    if s[0] != 1:
        x = np.reshape(x, (s[1],s[0]))
    px = poly_map(x,k)
    cval = px*pmean*px.T
    return cval[0,0]
    

stochastic_err=1e-3
delta=1e-9
sig0=1e-1
nx = 2
k = 5

feature_dim = int(sp.binom(nx+k,nx))
print('feature dimension: {:d}'.format(feature_dim))

nsamps_init = 1000
batchsize = 1000


# collect data, construct initial posterior

xs_raw = mksamps(nsamps_init, nx)
xs_normalized = (xs_raw - np.mean(xs_raw, axis=0)) / np.std(xs_raw,axis=0)
xs = poly_map(xs_normalized, k)
print(np.shape(xs))
print('done sampling')
#Lf, etaf, epsf = pacbayes_ellipse(xs, stochastic_err, delta, sig0)
Lf, etaf, epsf = pacbayes_ellipse(xs, stochastic_err, delta, sig0)
print('pacbayes err with initial sample size of {:d}: {:.3g}'.format(nsamps_init, epsf))
epss = [epsf]
nsamps_running = nsamps_init
targeteps = 0.05
#targeteps = 0.5
alpha=0.1

# keep collecting data and updating posterior until the accuracy criterion is
# met
while (1+alpha)*epsf > targeteps:
    nsamps_running += batchsize
    xsnew_raw = mksamps(batchsize, nx)
    xs_raw = np.vstack((xs_raw, xsnew_raw))
    xs_mean = np.mean(xs_raw, axis=0)
    xs_std = np.std(xs_raw, axis=0)
    xs_normalized = (xs_raw - xs_mean) / xs_std
    xs = poly_map(xs_normalized, k)
    print('done sampling')
    #Lf, etaf, epsf = pacbayes_ellipse(xs, stochastic_err, delta, sig0)
    Lf, etaf, epsf = pacbayes_ellipse(xs, stochastic_err, delta, sig0)
    epss.append(epsf)
    print('{:d} samples: {:.3g}'.format(nsamps_running, epsf))
print('')

beta = np.log((alpha+1)/(alpha))
etabloat = (1-2*np.sqrt(beta)/np.sqrt(nsamps_running))**(-1)
print('eta bloat factor: {:.3g}'.format(etabloat))

eps_apriori=.05
vcdim = int(sp.binom(nx+2*k,nx))
vcbound = math.ceil(5/eps_apriori*(np.log(4/delta)+vcdim*np.log(40/eps_apriori)))

print('Samples required for VC bound: {:d}'.format(vcbound))
print('Samples required by PAC-Bayes: {:d}'.format(nsamps_running))

vceta = -1
for x in xs:
    ellipsval = x * Lf * x.T
    if ellipsval > vceta:
        vceta = ellipsval[0,0]

vceta_pf = -1
for x_raw in xs_raw:
    x = poly_map(x_raw,k)
    ellipsval = x * Lf * x.T
    if ellipsval > vceta_pf:
        vceta_pf = ellipsval[0,0]

vceta_w = -1
for x_raw in xs_raw:
    ellipsval = cfun_wis(x_raw, Lf, k)
    if ellipsval > vceta_w:
        vceta_w = ellipsval

print('eta value for PAC-Bayes: {:.3g}'.format(etaf*etabloat))
print('eta value for PAC-VC: {:.3g}'.format(vceta))

# nsamps_post = 46052
# nout = 0
# xs_post = mksamps(nsamps_post, nx)
# for x in xs_post:
#     ellipsval = x*Lf*x.T
#     if ellipsval[0,0] > etaf*etabloat:
#         nout += 1
# 
# posterior_empirical_error = nout / nsamps_post
# 
# print('posterior sample error: {:.3g}'.format(posterior_empirical_error))


ngrid=100 
xgrid = np.linspace(-3,3,ngrid)
ygrid = np.linspace(-3,3,ngrid)
X,Y = np.meshgrid(xgrid,ygrid)
Z = np.zeros(np.shape(X))
Z_vc = np.zeros(np.shape(X))
for (i, xi) in enumerate(xgrid):
    for (j, yj) in enumerate(ygrid):
        v = np.matrix([[xi, yj]])
        cval = cfun_wis((v-xs_mean)/xs_std, Lf, k)
        Z[i,j] = cval - etaf*etabloat
        Z_vc[i,j] = cval - vceta

print(np.shape(xs))
plt.plot(xs_raw[:,1],xs_raw[:,0],linestyle='none', marker='.')
plt.contour(X,Y,Z, levels=[0], colors='green')
plt.contour(X,Y,Z_vc, levels=[0], colors='black', linestyles='dashed')
plt.show()
