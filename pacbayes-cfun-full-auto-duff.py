# external library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy.optimize as opt
from sampling import sample_oscillator_n
from time import perf_counter

# internal imports
from cfun import KernelCfun
from kernels import kfun_se

def pacbayes_cfun_iterative(sampler, eta, epsilon_target=.05, delta=1e-9, maxiter=1000, noiselevel=1e-3):
    # get the data dimension by taking a sample and examining its shape. We
    # expect the data to come in rows, so the second dimension will be the data
    # dimension.
    nx = np.shape(sampler(1))[1]
    cfun = KernelCfun(kfun=kfun, threshold=eta, noiselevel=noiselevel, delta=delta, nx=nx)
    i = 0
    epsilon_current=1
    print('iter, nsamps, kldiv, eps, time (s)')
    while epsilon_current >= epsilon_target and i <= maxiter:
        iter_time_start = perf_counter()
        #print('Computing samples')
        new_samples_raw = sampler(batchsize)
        cfun.add_data(new_samples_raw, normalize=True)
        
        # Bayesian PAC calculations (this is all contained in the KernelCfun class)
        epsilon_current = cfun.epsilon_pacbayes()

        iter_time_stop = perf_counter()
        iter_time = (iter_time_stop - iter_time_start)

        print('{:d}, {:d}, {:.5g},{:.3g},{:.5g}'.format(i, cfun.nsamps, cfun.kl_divergence, epsilon_current, iter_time))

        i = i + 1

    total_end = perf_counter()

    total_time = (total_end - total_start)
    print('total computation time: {:.5g} s'.format(total_time))

    return cfun

kfun = kfun_se
total_start = perf_counter()
batchsize=100
nx=2

eta = .25 # for SE kernel
sig0=eta/chi2.isf(.001,df=1)
stochastic_err = chi2.sf(eta/sig0, df=1)
print('Threshold: {:.3g}'.format(eta))
print('stochastic err: {:.3g}'.format(stochastic_err))
print('Noise level: {:.3g}'.format(sig0))

cfun = pacbayes_cfun_iterative(
        sampler=sample_oscillator_n,
        eta=eta,
        epsilon_target=0.2,
        noiselevel=sig0)

print('Plotting data and contour')
ngrid=20
gridwidth=3
xgrid = np.linspace(-gridwidth,gridwidth,ngrid)
ygrid = np.linspace(-gridwidth,gridwidth,ngrid)
X,Y = np.meshgrid(xgrid,ygrid)

Z = np.zeros(np.shape(X))
    
imat = np.eye(nsamps)
noisemat = np.matrix(sig0*np.eye(nsamps))
kmat = np.matrix(kfun(xs, xs))
postmat = np.linalg.inv(kmat + noisemat)

xin = np.array([]).reshape(0,2)
for (i, xi) in enumerate(xgrid):
    for (j, yj) in enumerate(ygrid):
        v = np.array([xi, yj])
        Z[i,j] = cfun.evaluate(v)

plt.plot(xs[:,1],xs[:,0],linestyle='none', marker='.')
plt.contour(X,Y,Z, levels=[eta], colors='green')

current_time = datetime.datetime.now()
timestr = current_time.strftime('%Y-%m-%d-%H%M')

plt.show()

