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

def pacbayes_cfun_iterative(cfun, sampler, epsilon_target, initialsamps, batchsize, maxsamps=100000, exact_stochastic_error=True):
    # get the data dimension by taking a sample and examining its shape. We
    # expect the data to come in rows, so the second dimension will be the data
    # dimension.
    total_start = perf_counter()
    i = 0
    epsilon_current=1
    logging.info('iter, nsamps, kldiv, eps, time (s)')
    while epsilon_current >= epsilon_target and i*batchsize <= maxsamps:
        iter_time_start = perf_counter()
        #logging.info('Computing samples')
        if i == 0:
            new_samples_raw = sampler(initialsamps)
        else:
            new_samples_raw = sampler(batchsize)
        cfun.add_data(new_samples_raw)
        # Bayesian PAC calculations (this is all contained in the KernelCfun class)
        # Try to apply a keyword to use the stochastic error upper bound if
        # possible. If not, just run the method directly.
        try:
            epsilon_current = cfun.epsilon_pacbayes(exact_stochastic_error=exact_stochastic_error)
        except TypeError:
            epsilon_current = cfun.epsilon_pacbayes()

        iter_time_stop = perf_counter()
        iter_time = (iter_time_stop - iter_time_start)
        logging.info('{:d}, {:d}, {:.5g},{:.3g},{:.5g}'.format(i, cfun.nsamps, cfun.kl_divergence, epsilon_current, iter_time))
        i = i + 1
    total_end = perf_counter()
    total_time = (total_end - total_start)
    logging.info('total computation time: {:.5g} s'.format(total_time))
    return None


def comparison_experiment_local(sampler, nx, epsilon_target, delta, experiment_name='exp', k=10):

    targeteps = 0.05

    #k=10
    sig0=1e-3
    nf = int(sp.binom(nx + k, nx))  # number of features (i.e. monomials)
    threshold_poly = nf / targeteps # based on Markov inequality estimate for (1-eps) threshold
    cfun_poly = PolyCfun(order=k, threshold=threshold_poly, noiselevel=sig0, delta=delta, nx=2)

    threshold_se = 0.25
    noiselevel_se=threshold_se/chi2.isf(.001,df=1)
    #cfun_se = KernelCfun(kfun=kfun_se, threshold=threshold_se, noiselevel=noiselevel_se, delta=delta, nx=2)

    cfun_nys = KernelCfun(kfun=kfun_se, threshold=threshold_se, noiselevel=noiselevel_se, delta=delta, nx=2, n_nys=10000)


    logging.info('squared exponential (Nystrom)')
    pacbayes_cfun_iterative(
        cfun=cfun_nys,
        sampler=sampler,
        epsilon_target=epsilon_target,
        initialsamps=12000,
        batchsize=4000,
        maxsamps=100000,
        exact_stochastic_error=False)

    logging.info('polynomial')
    pacbayes_cfun_iterative(
        cfun=cfun_poly,
        sampler=sampler,
        epsilon_target=epsilon_target,
        initialsamps=2000,
        batchsize=1000,
        maxsamps=100000,
        exact_stochastic_error=False)

    # logging.info('squared exponential (full)')
    # pacbayes_cfun_iterative(
    #     cfun=cfun_se,
    #     sampler=sampler,
    #     epsilon_target=epsilon_target,
    #     initialsamps=12000,
    #     batchsize=2000,
    #     maxsamps=20000,
    #     exact_stochastic_error=False)

    logging.info('Plotting data and contour')

    xrange, yrange = data_plot_limits(cfun_poly.data_raw, padfactor=0.2)

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

    cfun_poly_evals = cfun_poly.evaluate(xin)
    Z_poly = cfun_poly_evals.reshape(ngrid,ngrid).T

    #cfun_se_evals = cfun_se.evaluate(xin)
    #Z_se = cfun_se_evals.reshape(ngrid,ngrid).T

    cfun_nys_evals = cfun_nys.evaluate(xin)
    Z_nys = cfun_nys_evals.reshape(ngrid,ngrid).T

    plt.clf()
    plt.plot(cfun_poly.data_raw[:,0],cfun_poly.data_raw[:,1],linestyle='none', marker='.')
    plt.contour(X,Y,Z_poly, levels=[threshold_poly], colors='green', label='Polynomial')
    #plt.contour(X,Y,Z_se, levels=[threshold_se], colors='red', label='SE')
    plt.contour(X,Y,Z_nys, levels=[threshold_se], colors='blue', label='SE Nystrom')

    current_time = datetime.datetime.now()
    timestr = current_time.strftime('%Y-%m-%d-%H%M')
    figname = './results/{:s}_{:s}_cmp.png'.format(timestr, experiment_name)
    plt.savefig(figname, dpi=300)


# configure logging
current_time = datetime.datetime.now()
timestr = current_time.strftime('%Y-%m-%d-%H%M')
logging.basicConfig(
        handlers=[
            logging.FileHandler("./results/{:s}-experiments-savio.log".format(timestr)),
            logging.StreamHandler()
        ],
        format='%(asctime)s: %(message)s',
        level=logging.INFO)

epsilon_target = 0.05

logging.info('Experiment #1: Duffing oscillator')
comparison_experiment_local(sampler=sample_oscillator_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        experiment_name='duffing')

logging.info('Experiment #2: Quadrotor (x,h)')
comparison_experiment_local(sampler=sample_quadrotor_xh_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        experiment_name='quadrotor', k=4)

logging.info('Experiment #3: Traffic (last 2)')
comparison_experiment_local(sampler=sample_traffic_end_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        experiment_name='traffic')


