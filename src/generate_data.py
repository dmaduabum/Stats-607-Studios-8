import numpy as np
from scipy import stats

def ar1_cov(p, rho):
    """
    AR(1) covariance with entries rho^{|i - j|}
    """
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])

def sample_X(n, p, rho, rng):
    """
    Gaussian design with AR(1) covariance.
    """

    Sigma = ar1_cov(p, rho)
    L = np.linalg.cholesky(Sigma)
    normal_mat = rng.standard_normal(size = (n, p))
    return normal_mat @ L.T

def make_beta(p, rng):
    """
    Create coefficient vector
    """
    b = rng.standard_normal(p)
    return b

def sigma_for_SNR(X, beta, target_snr):
    """
    Choose sigma so SNR = Var(X beta) / sigma^2 approximately equals target_snr
    """

    s2 = np.var(X @ beta)
    sigma2 = s2 / float(target_snr)
    return np.sqrt(sigma2)


def sample_errors(n, df, sigma, rng):
    """
    Student-t (df) errors scaled to sigma.
    """

    if np.isinf(df):
        err = rng.standard_normal(n) * sigma
        return err

    t = stats.t.rvs(df, size = n, random_state=rng)
    if df > 2:
        scale = sigma / np.sqrt(df / (df - 2))
    else:
        scale = sigma / (np.std(t))
    
    return t * scale


def simulate_dataset(n, gamma, rho, df, snr, seed = None):

    if seed is None:
        rng = np.random.default_rng(82803)
    else:
        rng = np.random.default_rng(seed)
    
    p = int(round(gamma * n))

    X = sample_X(n, p, rho, rng)
    beta = make_beta(p, rng)
    sigma = sigma_for_SNR(X, beta, target_snr=snr)
    eps = sample_errors(n, df=df, sigma=sigma, rng=rng)
    y = X @ beta + eps

    return {
        "X": X,
        "y": y,
        "beta": beta,
        "sigma": sigma,
        "params": {"n": n, "p": p, "gamma": gamma, "rho": rho, 
                   "df": df, "snr": snr, "seed": seed}
    }