import numpy as np
import scipy
from scipy.stats import chi2
import scipy.optimize as opt
import scipy.special as sp
from sklearn.preprocessing import PolynomialFeatures


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
    s, ld0 = np.linalg.slogdet(cov0)
    s, ld1 = np.linalg.slogdet(cov1)
    cov1_inv = np.linalg.inv(cov1)
    tr = np.trace(np.matmul(cov1_inv, cov0))
    kl = 0.5 * (ld1 - ld0 + tr - n)
    return kl


class KernelCfun:
    def __init__(self, kfun, threshold, noiselevel, delta, nx, n_nys=-1):
        self.kfun = kfun
        self.noiselevel = noiselevel
        self.delta = delta
        self.nx = nx
        self.data = np.array([]).reshape(0,nx)
        self.data_raw = np.array([]).reshape(0,nx)
        self.epsilon = 1
        self.threshold = threshold
        self.n_nys = n_nys
        self.raw_data_mean = 0
        self.raw_data_std = 1


    def add_data(self, newdata):
        self.data = np.concatenate((self.data, newdata))
        self.data_raw = np.concatenate((self.data_raw, newdata))
        self.nsamps = np.shape(self.data)[0]
        dmean = np.mean(self.data_raw, axis=0)
        dstd = np.std(self.data_raw, axis=0)
        self.data = (self.data_raw - dmean) / dstd
        self.raw_data_mean = dmean
        self.raw_data_std = dstd
        return None

    
    def compute_kl_divergence(self):
        if self.n_nys > 0 and self.nsamps > self.n_nys:
            kl = self.compute_kl_divergence_nys()
        else:
            kl = self.compute_kl_divergence_full()
        return kl

    def compute_empirical_stochastic_risk(self):
        if self.n_nys > 0 and self.nsamps > self.n_nys:
            kl = self.compute_empirical_stocahstic_risk_nys()
        else:
            kl = self.compute_empirical_stochastic_risk_full()
        return kl

    def evaluate(self, xin):
        if self.n_nys > 0 and self.nsamps > self.n_nys:
            kl = self.evaluate_nys(xin)
        else:
            kl = self.evaluate_full(xin)
        return kl

    def compute_kl_divergence_full(self):
        # construct kernel matrix
        sig0 = self.noiselevel
        imat = np.eye(self.nsamps)
        noisemat = np.matrix(sig0*np.eye(self.nsamps))
        kmat = np.matrix(self.kfun(self.data, self.data))
        #Compute KL divergence
        evec = np.linalg.eigvalsh(kmat)
        ld = np.sum(np.log(1+sig0**(-1)*evec))
        tr = np.sum((1 + sig0**(-1)*evec)**(-1)) - self.nsamps
        kl = 0.5*(ld + tr)
        self.kl_divergence = kl
        return kl


    def compute_kl_divergence_nys(self, kmm_perturbation=1e-6):
        kmm = self.kfun(self.data[:self.n_nys, :], self.data[:self.n_nys, :])
        # For some kernels, kmm may not be positive definite due to numerical
        # instabilities. It's necessary for kmm to be numerically positive
        # definite in the generalized eigenvalue problem, so we add a small
        # positive perturbation.
        kmm = kmm + kmm_perturbation*np.eye(self.n_nys)
        kmn = self.kfun(self.data[:self.n_nys, :], self.data)
        kmnnm = np.matmul(kmn, kmn.T)
        evals_nys = scipy.linalg.eigvalsh(np.matmul(kmn,kmn.T), kmm)
        evec = np.concatenate((evals_nys, np.zeros((self.nsamps-self.n_nys,))))
        ld = np.sum(np.log(1+self.noiselevel**(-1)*evec))
        tr = np.sum((1 + self.noiselevel**(-1)*evec)**(-1)) - self.nsamps
        kl = 0.5*(ld + tr)
        self.kl_divergence = kl
        return kl


    def epsilon_pacbayes(self, recompute_kl=True, exact_stochastic_error=True):
        if recompute_kl:
            kl = self.compute_kl_divergence()
            self.kl_divergence = kl
        else:
            kl = self.kl_divergence

        if exact_stochastic_error:
            stochastic_err = self.compute_empirical_stochastic_risk()
        else:
            stochastic_err = chi2.sf(self.threshold / self.noiselevel, 1)

        pbrhs = (kl + np.log((self.nsamps + 1)/self.delta)) / self.nsamps
        errbnd_stochastic = klber_ub(stochastic_err, pbrhs)
        errbnd_mean = errbnd_stochastic / chi2.cdf(1,1)
        return errbnd_mean


    def compute_empirical_stochastic_risk_full(self):
        noisemat = np.matrix(self.noiselevel*np.eye(self.nsamps))
        kmat = np.matrix(self.kfun(self.data, self.data))
        postcov = kmat - kmat*np.linalg.inv(kmat + noisemat)*kmat
        cfun_evals = np.diag(postcov)
        point_errors = chi2.sf(self.threshold / cfun_evals, 1)
        stochastic_err = np.mean(point_errors)
        return stochastic_err


    def compute_empirical_stochastic_risk_nys(self):
        cfun_evals = self.evaluate_nys(self.data)
        point_errors = chi2.sf(self.threshold / cfun_evals, 1)
        stochastic_err = np.mean(point_errors)
        return stochastic_err


    def evaluate_full(self, xin):
        xindim = np.shape(xin)[0]
        noisemat = np.matrix(self.noiselevel*np.eye(self.nsamps))
        kmat = np.matrix(self.kfun(self.data, self.data))
        postmat = np.linalg.inv(kmat + noisemat)
        cfun_evals = np.zeros((xindim,))
        for (i, xini) in enumerate(xin):
            v = (xini - self.raw_data_mean) / self.raw_data_std
            kxx = self.kfun(v,v)
            kvec = self.kfun(self.data, v)
            kv = np.matmul(kvec.T,np.matmul(postmat,kvec))
            cfun_evals[i] = kxx[0,0] - kv[0,0]
        return cfun_evals


    def evaluate_nys(self, xin, kmm_perturbation=1e-3):
        xindim = np.shape(xin)[0]
        kmm = self.kfun(self.data[:self.n_nys, :], self.data[:self.n_nys, :])
        kmm = kmm + kmm_perturbation*np.eye(self.n_nys)
        kmn = self.kfun(self.data[:self.n_nys, :], self.data)
        kmnnm = np.matmul(kmn, kmn.T)
        kinv = np.linalg.inv(self.noiselevel*kmm + kmnnm)
        cfun_evals = np.zeros((xindim,))
        for (i, xini) in enumerate(xin):
            v = (xini - self.raw_data_mean) / self.raw_data_std
            kxx = self.kfun(v,v)
            knx = self.kfun(self.data,v)
            kxnnx = np.matmul(knx.T, knx)
            knxnm = np.matmul(knx.T, kmn.T)
            kv = self.noiselevel**(-1)*np.matmul(knxnm, np.matmul(kinv, knxnm.T))
            zval = kxx - self.noiselevel**(-1)*kxnnx + kv
            cfun_evals[i] = zval[0,0]
        return cfun_evals


class PolyCfun:
    def __init__(self, order, threshold, noiselevel, delta, nx):
        self.order = order
        self.feature_dimension = int(sp.binom(nx + order, nx))
        self.noiselevel = noiselevel
        self.delta = delta
        self.nx = nx
        self.data = np.array([]).reshape(0,nx)
        self.data_raw = np.array([]).reshape(0,nx)
        self.epsilon = 1
        self.threshold = threshold
        self.raw_data_mean = 0
        self.raw_data_std = 1

    def add_data(self, newdata, normalize=False):
        self.data = np.concatenate((self.data, newdata))
        self.data_raw = np.concatenate((self.data_raw, newdata))
        self.nsamps = np.shape(self.data)[0]
        dmean = np.mean(self.data_raw, axis=0)
        dstd = np.std(self.data_raw, axis=0)
        self.data = (self.data_raw - dmean) / dstd
        self.raw_data_mean = dmean
        self.raw_data_std = dstd
        return None

    def compute_kl_divergence(self):
        pf = PolynomialFeatures(self.order)
        zxs = pf.fit_transform(self.data)

        empirical_moment_matrix = self.noiselevel * np.eye(self.feature_dimension) + np.matmul(zxs.T,zxs) / self.nsamps
        prior_matrix = self.noiselevel**(-1) * np.eye(self.feature_dimension)
        inverse_empirical_moment_matrix = np.linalg.inv(empirical_moment_matrix)

        kl = kl_centered_normal(inverse_empirical_moment_matrix, prior_matrix)

        return kl

    def classical_pac_sample_size(self, eps):
        return None

    def epsilon_pacbayes(self, recompute_kl=True):

        pf = PolynomialFeatures(self.order)
        zxs = pf.fit_transform(self.data)

        empirical_moment_matrix = self.noiselevel * np.eye(self.feature_dimension) + np.matmul(zxs.T,zxs) / self.nsamps
        inverse_empirical_moment_matrix = np.linalg.inv(empirical_moment_matrix)
        prior_matrix = self.noiselevel**(-1) * np.eye(self.feature_dimension)

        cfun_evals = np.zeros((self.nsamps,))
        for (i, zxi) in enumerate(zxs):
            cfun_evals[i] = np.matmul(zxi, np.matmul(inverse_empirical_moment_matrix, zxi.T))

        point_errors = chi2.sf(self.threshold / cfun_evals, 1)

        stochastic_err = np.mean(point_errors)

        kl = kl_centered_normal(inverse_empirical_moment_matrix, prior_matrix)
        self.kl_divergence = kl

        pbrhs = (kl + np.log((self.nsamps + 1)/self.delta)) / self.nsamps

        point_errors = chi2.sf(self.threshold / cfun_evals, 1)
        stochastic_err = np.mean(point_errors)

        errbnd_stochastic = klber_ub(stochastic_err, pbrhs)
        errbnd_mean = errbnd_stochastic / chi2.cdf(1,1)
        return errbnd_mean


    def evaluate(self, xin):
        xindim = np.shape(xin)[0]
        pf = PolynomialFeatures(self.order)
        zxs = pf.fit_transform(self.data)
        xin_normalized = (xin - self.raw_data_mean) / self.raw_data_std
        zxin = pf.fit_transform(xin_normalized)

        empirical_moment_matrix = self.noiselevel * np.eye(self.feature_dimension) + np.matmul(zxs.T,zxs) / self.nsamps
        inverse_empirical_moment_matrix = np.linalg.inv(empirical_moment_matrix)

        cfun_evals = np.zeros((xindim,))
        for (i, zxi) in enumerate(zxin):
            cfun_evals[i] = np.matmul(zxi, np.matmul(inverse_empirical_moment_matrix, zxi.T))

        return cfun_evals


