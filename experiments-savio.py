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

def classical_pac(cfun, sampler, epsilon_target, delta):
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
        logging.info('done sampling')
        # Bayesian PAC calculations (this is all contained in the KernelCfun class)
        # Try to apply a keyword to use the stochastic error upper bound if
        # possible. If not, just run the method directly.
        try:
            epsilon_current = cfun.epsilon_pacbayes(i+1,exact_stochastic_error=exact_stochastic_error)
        except TypeError:
            epsilon_current = cfun.epsilon_pacbayes(i+1)

        iter_time_stop = perf_counter()
        iter_time = (iter_time_stop - iter_time_start)
        logging.info('{:d}, {:d}, {:.5g},{:.3g},{:.5g}'.format(i, cfun.nsamps, cfun.kl_divergence, epsilon_current, iter_time))
        i = i + 1
    total_end = perf_counter()
    total_time = (total_end - total_start)
    logging.info('total computation time: {:.5g} s'.format(total_time))
    return None


def comparison_experiment(sampler, nx, epsilon_target, delta, experiment_name='exp', k=10, kfun=kfun_se, threshold_kernel=0.25, xname='$x$', yname='$y$'):

    targeteps = 0.05

    #k=10
    sig0=1e-3
    nf = int(sp.binom(nx + k, nx))  # number of features (i.e. monomials)
    threshold_poly = nf / targeteps # based on Markov inequality estimate for (1-eps) threshold
    cfun_poly = PolyCfun(order=k, threshold=threshold_poly, noiselevel=sig0, delta=delta, nx=2)
    cfun_poly_classical = PolyCfun(order=k, threshold=threshold_poly, noiselevel=sig0, delta=delta, nx=2)

    noiselevel_se=threshold_kernel/chi2.isf(.001,df=1)
    cfun_se = KernelCfun(kfun=kfun, threshold=threshold_kernel, noiselevel=noiselevel_se, delta=delta, nx=2)

    logging.info('polynomial (classical)')
    pacbayes_cfun_classical(cfun_poly_classical, sampler, epsilon_target)
    classical_evals = cfun_poly_classical.evaluate(cfun_poly_classical.data_raw)
    classical_max_eval = np.max(classical_evals)
    logging.info('classical max cfun value: {:.5f}'.format(classical_max_eval))

    logging.info('polynomial')
    pacbayes_cfun_iterative(
        cfun=cfun_poly,
        sampler=sampler,
        epsilon_target=epsilon_target,
        initialsamps=2000,
        batchsize=1000,
        maxsamps=100000,
        exact_stochastic_error=False)


    logging.info('squared exponential (full)')
    pacbayes_cfun_iterative(
        cfun=cfun_se,
        sampler=sampler,
        epsilon_target=epsilon_target,
        initialsamps=20000,
        batchsize=5000,
        maxsamps=200000,
        exact_stochastic_error=False)


    logging.info('Computing a posteriori accuracy:')

    xin_ap = sampler(46052)

    cfun_poly_evals_ap = cfun_poly.evaluate(xin_ap)
    ap_error_poly = np.mean(np.where(cfun_poly_evals_ap > threshold_poly,1,0))
    cfun_poly_classical_evals_ap = cfun_poly_classical.evaluate(xin_ap)
    ap_error_classical = np.mean(np.where(cfun_poly_classical_evals_ap > classical_max_eval,1,0))
    cfun_nys_evals_ap = cfun_se.evaluate(xin_ap)
    ap_error_se = np.mean(np.where(cfun_nys_evals_ap > threshold_kernel,1,0))
    logging.info('    poly (kernel)   : {:f}'.format(ap_error_poly))
    logging.info('    poly (classical): {:f}'.format(ap_error_classical))
    logging.info('    SE   (kernel)   : {:f}'.format(ap_error_se))


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

    
    eval_time_poly_start = perf_counter()
    cfun_poly_evals = cfun_poly.evaluate(xin)
    Z_poly = cfun_poly_evals.reshape(ngrid,ngrid).T
    eval_time_poly_end = perf_counter()
    eval_time_poly = eval_time_poly_end - eval_time_poly_start
    logging.info('time to evaluate poly: {:f}'.format(eval_time_poly))

    eval_time_poly_classical_start = perf_counter()
    cfun_poly_classical_evals = cfun_poly_classical.evaluate(xin)
    Z_poly_classical = cfun_poly_classical_evals.reshape(ngrid,ngrid).T
    eval_time_poly_classical_end = perf_counter()
    eval_time_poly_classical = eval_time_poly_classical_end - eval_time_poly_classical_start
    logging.info('time to evaluate poly (classical): {:f}'.format(eval_time_poly_classical))

    eval_time_se_start = perf_counter()
    cfun_se.n_nys = -1
    cfun_se_evals = cfun_se.evaluate(xin)
    Z_se = cfun_se_evals.reshape(ngrid,ngrid).T
    eval_time_se_end = perf_counter()
    eval_time_se = eval_time_se_end - eval_time_se_start
    logging.info('time to evaluate se: {:f}'.format(eval_time_se))

    eval_time_nys_start = perf_counter()
    cfun_se.n_nys = 2000
    cfun_nys_evals = cfun_se.evaluate(xin)
    Z_nys = cfun_nys_evals.reshape(ngrid,ngrid).T
    eval_time_nys_end = perf_counter()
    eval_time_nys = eval_time_nys_end - eval_time_nys_start
    logging.info('time to evaluate nys: {:f}'.format(eval_time_nys))

    plt.clf()
    plt.plot(cfun_poly.data_raw[:,0],cfun_poly.data_raw[:,1],linestyle='none', marker='.')
    plt.contour(X,Y,Z_poly, levels=[threshold_poly], colors='green', label='Polynomial')
    plt.contour(X,Y,Z_poly_classical, levels=[classical_max_eval], colors='black', label='Polynomial (classical)')
    plt.contour(X,Y,Z_se, levels=[threshold_kernel], colors='blue', label='SE')
    #plt.contour(X,Y,Z_nys, levels=[threshold_kernel], colors='red', label='SE Nystrom')
    plt.xlabel(xname)
    plt.ylabel(yname)

    current_time = datetime.datetime.now()
    timestr = current_time.strftime('%Y-%m-%d-%H%M')
    figname = './results/{:s}_{:s}_cmp.png'.format(timestr, experiment_name)
    plt.savefig(figname, dpi=300)


# configure logging
current_time = datetime.datetime.now()
timestr = current_time.strftime('%Y-%m-%d-%H%M')
logging.basicConfig(
        handlers=[
            #logging.FileHandler("./results/{:s}-experiments-savio.log".format(timestr)),
            logging.StreamHandler()
        ],
        format='%(asctime)s: %(message)s',
        level=logging.INFO)

#epsilon_target = 0.05
epsilon_target = 0.10

from functools import partial

logging.info('Experiment #1: Duffing oscillator')

kfun_duff = partial(kfun_se, sig=1/4)
comparison_experiment(sampler=sample_oscillator_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        kfun=kfun_duff,
        experiment_name='duffing',
        xname='$z$', yname='$y$')

kfun_quad = partial(kfun_se, sig=1/4)
logging.info('Experiment #2: Quadrotor (x,h)')
comparison_experiment(sampler=sample_quadrotor_xh_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        kfun=kfun_quad,
        experiment_name='quadrotor', k=4,
        xname='$p_x$', yname='$p_h$')

kfun_traf = partial(kfun_se, sig=1/4)
logging.info('Experiment #3: Traffic (last 2)')
comparison_experiment(sampler=sample_traffic_end_n,
        nx=2,
        epsilon_target=epsilon_target,
        delta=1e-9,
        kfun=kfun_traf,
        experiment_name='traffic',
        xname='$x_5$', yname='$x_6$')


