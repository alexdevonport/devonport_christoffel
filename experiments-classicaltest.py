# external library imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy.optimize as opt
from time import perf_counter
import scipy.special as sp
import datetime
import logging

# internal imports
from cfun import KernelCfun, PolyCfun
from kernels import kfun_se
from sampling import (sample_oscillator_n, 
        sample_quadrotor_n,
        sample_quadrotor_xh_n,
        sample_traffic_n,
        sample_traffic_end_n)

def classical_pac(cfun, sampler, epsilon_targer, delta):
    return None

def data_plot_limits(data, padfactor=0.1):
    dmean = np.mean(data, axis=0)
    drange = np.ptp(data, axis=0) # PTP is short for "peak to peak" apparently
    xrange = np.array( [dmean[0] - 0.5*(1+padfactor)*drange[0], dmean[0] + 0.5*(1+padfactor)*drange[0]])
    yrange = np.array( [dmean[1] - 0.5*(1+padfactor)*drange[1], dmean[1] + 0.5*(1+padfactor)*drange[1]])
    return xrange, yrange

def pacbayes_cfun_classical(cfun, sampler, epsilon_target):
    total_start = perf_counter()
    classical_sample_bound = cfun.classical_sample_bound(epsilon_target)
    logging.info('classical requires {:d} samples'.format(classical_sample_bound))
    new_samples_raw = sampler(classical_sample_bound)
    cfun.add_data(new_samples_raw)
    total_end = perf_counter()
    total_time = (total_end - total_start)
    logging.info('total computation time: {:.5g} s'.format(total_time))
    return None

def comparison_experiment(sampler, nx, epsilon_target, delta, experiment_name='exp', k=10):

    targeteps = 0.05

    #k=10
    sig0=1e-3
    nf = int(sp.binom(nx + k, nx))  # number of features (i.e. monomials)
    threshold_poly = nf / targeteps # based on Markov inequality estimate for (1-eps) threshold
    cfun_poly_classical = PolyCfun(order=k, threshold=threshold_poly, noiselevel=sig0, delta=delta, nx=2)


    logging.info('polynomial (classical)')
    pacbayes_cfun_classical(cfun_poly_classical, sampler, epsilon_target)

    logging.info('Plotting data and contour')

    xrange, yrange = data_plot_limits(cfun_poly_classical.data_raw, padfactor=0.2)

    ngrid=100
    gridwidth=3
    xgrid = np.linspace(xrange[0],xrange[1],ngrid)
    ygrid = np.linspace(yrange[0], yrange[1],ngrid)
    X,Y = np.meshgrid(xgrid,ygrid)

    xin = np.array([]).reshape(0,2)
    for (i, xi) in enumerate(xgrid):
        for (j, yj) in enumerate(ygrid):
            v = np.array([[xi, yj]])
            xin = np.concatenate((xin, v))

    cfun_poly_classical_evals = cfun_poly_classical.evaluate(xin)
    Z_poly_classical = cfun_poly_classical_evals.reshape(ngrid,ngrid).T

    plt.clf()
    plt.plot(cfun_poly_classical.data_raw[:,0],cfun_poly_classical.data_raw[:,1],linestyle='none', marker='.')
    plt.contour(X,Y,Z_poly_classical, levels=[threshold_poly], colors='black', label='Polynomial (classical)')

    current_time = datetime.datetime.now()
    timestr = current_time.strftime('%Y-%m-%d-%H%M')
    figname = './results/{:s}_{:s}_cmp.png'.format(timestr, experiment_name)
    plt.savefig(figname, dpi=300)


# configure logging
current_time = datetime.datetime.now()
timestr = current_time.strftime('%Y-%m-%d-%H%M')
logging.basicConfig(
        handlers=[
            logging.FileHandler("./results/{:s}-experiments-classicaltest.log".format(timestr)),
            logging.StreamHandler()
        ],
        format='%(asctime)s: %(message)s',
        level=logging.INFO)

epsilon_target = 0.75

logging.info('Experiment #1: Duffing oscillator')
comparison_experiment(sampler=sample_oscillator_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        experiment_name='duffing')

