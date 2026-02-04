import os
import numpy as np
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import chi2

c_light = 299792.458  # km/s

def inv(K, jitter=1e-10):
    # K = 0.5 * (K + K.T) 
    # L = np.linalg.cholesky(K + jitter * np.eye(K.shape[0]))
    # Linv = np.linalg.solve(L, np.eye(K.shape[0]))
    # K_inv = Linv.T @ Linv
    # return (K_inv)
    K = 0.5 * (K + K.T)
    L = np.linalg.cholesky(K + jitter*np.eye(K.shape[0]))
    Linv = np.linalg.solve(L, np.eye(K.shape[0]))
    return Linv.T @ Linv

###############################################################################
# 1.  mB(z) reconstruction utilities
###############################################################################
def evaluate_mB(z, mB_nodes, z_nodes):
    """Interpolate mB(z) using predefined node values."""
    interp = interp1d(np.log(z_nodes), mB_nodes, kind='linear', fill_value="extrapolate")
    return interp(np.log(z))


def ln_likelihood(mB_nodes, z, m_obs, cov_inv, z_nodes):
    m_model = evaluate_mB(z, mB_nodes, z_nodes)
    delta = m_obs - m_model
    return -0.5 * delta.T @ cov_inv @ delta


def ln_prior(mB_nodes):
    if np.all((10 < mB_nodes) & (mB_nodes < 30)):
        return 0.0
    return -np.inf


def ln_posterior(mB_nodes, z, m_obs, cov_inv, z_nodes):
    lp = ln_prior(mB_nodes)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_likelihood(mB_nodes, z, m_obs, cov_inv, z_nodes)


###############################################################################
# 2. BAO: DV→DA→DL or DM→DA→DL
###############################################################################
def bao_DV_to_DL(z, DV_rd, DV_rd_err, H, H_err, rd, rd_err):
    """Convert DV/rd data to DL."""
    DA_rd = DV_rd**1.5 * rd**0.5 * H**0.5 / (c_light**0.5 * z**0.5 * (1 + z))

    DA_rd_err = DA_rd * np.sqrt(
        (1.5 * DV_rd_err / DV_rd)**2 +
        (0.5 * rd_err / rd)**2 +
        (0.5 * H_err / H)**2
    )

    DA = DA_rd * rd
    DA_err = DA * np.sqrt((DA_rd_err / DA_rd)**2 + (rd_err / rd)**2)

    DL = DA * (1 + z)**2
    DL_err = DA_err * (1 + z)**2
    return DL, DL_err


def bao_DM_to_DL(z, DM_rd, DM_rd_err, rd, rd_err):
    """Convert DM/rd to DL."""
    DA_rd = DM_rd / (1 + z)
    DA_rd_err = DM_rd_err / (1 + z)

    DA = DA_rd * rd
    DA_err = DA * np.sqrt((DA_rd_err / DA_rd)**2 + (rd_err / rd)**2)

    DL = DA * (1 + z)**2
    DL_err = DA_err * (1 + z)**2
    return DL, DL_err


###############################################################################
# 3. Gaussian Process reconstruction for BAO DL
###############################################################################
def fit_gp_DL(z, DL, DL_err):
    kernel = ConstantKernel(1.0, (1e-3, 1e10)) * Matern(length_scale=0.1, nu=5/2)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=DL_err**2, n_restarts_optimizer=10)
    gp.fit(z.reshape(-1,1), DL)
    return gp


###############################################################################
# 4. SN DL from mB(z) samples
###############################################################################
def compute_DL_SN(z_eval, mB_nodes, MB_sample, z_nodes):
    mB_eval = evaluate_mB(z_eval, mB_nodes, z_nodes)
    mu = mB_eval - MB_sample
    return 10**((mu - 25)/5)


###############################################################################
# 5. DDR eta(z) calculation
###############################################################################
def compute_eta_samples(DL_sn_samples, gp, z_eval, n_samples):
    DL_bao_samples = gp.sample_y(z_eval.reshape(-1,1), n_samples).T
    eta_samples = DL_sn_samples / DL_bao_samples
    return eta_samples


###############################################################################
# 6. χ² test at BAO points
###############################################################################
def eta_chi2(eta_bao, eta_bao_err):
    chi2_val = np.sum((eta_bao - 1)**2 / eta_bao_err**2)
    p_val = 1 - chi2.cdf(chi2_val, df=len(eta_bao))
    return chi2_val, p_val
###############################################################################

def load_bao_data(project_root=None):
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")

    file_DV = os.path.join(data_dir, "Combine_DV_rd.dat")
    file_DM = os.path.join(data_dir, "Combine_DM_rd.dat")
    file_Hz = os.path.join(output_dir, "Hz_DVreconstruct_results.txt")

    z_DV, DV_rd, DV_rd_err = np.loadtxt(file_DV, unpack=True)
    z_DM, DM_rd, DM_rd_err = np.loadtxt(file_DM, unpack=True)
    _, H_DV, H_DV_err = np.loadtxt(file_Hz, unpack=True)

    return {
        "DV": {"z": z_DV, "value": DV_rd, "error": DV_rd_err},
        "DM": {"z": z_DM, "value": DM_rd, "error": DM_rd_err},
        "Hz": {"z": None, "value": H_DV, "error": H_DV_err},
    }

def calucate_AIC(n_para, log_probs):
    max_log_L = np.max(log_probs)
    aic = 2 * n_para - 2 * max_log_L
    return aic

def calucate_BIC(n_data, n_para, log_probs):
    max_log_L = np.max(log_probs)
    bic = n_para * np.log(n_data) - 2 * max_log_L
    return bic

def read_cov(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    N_cov = int(lines[0].strip())
    cov_vec = np.array([float(x.strip()) for x in lines[1:]])
    cov_full = cov_vec.reshape((N_cov, N_cov))
    return cov_full

# def matern_kernel(x1, x2, l, sf, nu):
#     d = np.abs(x1[:, None] - x2[None, :])
#     if nu == 1.5:
#         return sf**2 * (1 + np.sqrt(3)*d/l) * np.exp(-np.sqrt(3)*d/l)
#     elif nu == 2.5:
#         sqrt5 = np.sqrt(5)
#         return sf**2 * (1 + sqrt5*d/l + 5*d**2/(3*l**2)) * np.exp(-sqrt5*d/l)
#     return sf**2 * np.exp(-0.5*(d/l)**2)

# def log_posterior(theta, x, y, cov, nu):
#     log_l, log_sf = theta
#     if not (-5 < log_l < 5) or not (-5 < log_sf < 10):
#         return -np.inf
#     l, sf = np.exp(log_l), np.exp(log_sf)
    
#     K = matern_kernel(x, x, l, sf, nu)
#     K += cov + np.eye(len(x)) * 1e-6
    
#     L = np.linalg.cholesky(K)
#     alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
#     return -0.5 * y @ alpha - np.sum(np.log(np.diag(L)))



def matern_kernel(x1, x2, theta, nu=2.5):
    l, sigma_f = theta
    x1 = np.atleast_2d(x1).T if np.ndim(x1) == 1 else np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)
    d = np.abs(x1 - x2)
    if np.isclose(nu, 0.5):
        K = sigma_f**2 * np.exp(-d / l)
    elif np.isclose(nu, 1.5):
        sqrt3 = np.sqrt(3.0)
        K = sigma_f**2 * (1 + sqrt3 * d / l) * np.exp(-sqrt3 * d / l)
    elif np.isclose(nu, 2.5):
        sqrt5 = np.sqrt(5.0)
        K = sigma_f**2 * (
            1 + sqrt5 * d / l + 5.0 * d**2 / (3.0 * l**2)
        ) * np.exp(-sqrt5 * d / l)
    else:
        from scipy.special import gamma, kv
        d_safe = np.maximum(d, 1e-12)
        arg = np.sqrt(2 * nu) * d_safe / l
        factor = (2**(1. - nu)) / gamma(nu)
        K = sigma_f**2 * factor * (arg**nu) * kv(nu, arg)
        K[d == 0.0] = sigma_f**2
    return K

def rbf_kernel(x1, x2, theta):
    l, sigma_f = theta
    x1 = np.atleast_2d(x1).T if np.ndim(x1) == 1 else np.atleast_2d(x1)
    x2 = np.atleast_2d(x2).T if np.ndim(x2) == 1 else np.atleast_2d(x2)
    d = np.abs(x1 - x2.T)
    K = sigma_f**2 * np.exp(-0.5 * (d / l)**2)
    return K

def gaussian_kernel_vectorization(x1, x2, theta):
    """More efficient approach."""
    l, sigma_f = theta
    # 保证输入是二维
    if x1.ndim == 1:
        x1 = x1[:, None]
    if x2.ndim == 1:
        x2 = x2[:, None]
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + \
                np.sum(x2**2, 1) - \
                2 * np.dot(x1, x2.T)
    K = sigma_f**2 * np.exp(-0.5 / l**2 * dist_matrix)
    return K

def gp_predict(x_obs, y_obs, cov_obs, x_pred, kernel_func, theta, **kwargs):
    K = kernel_func(x_obs, x_obs, theta, **kwargs)
    K_s = kernel_func(x_pred, x_obs, theta, **kwargs)
    K_ss = kernel_func(x_pred, x_pred, theta, **kwargs)

    A = K + cov_obs
    A += 1e-8 * np.eye(len(A))
    try:
        L = np.linalg.cholesky(A)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
        v = np.linalg.solve(L, K_s.T)
        y_mean = K_s @ alpha
        y_cov = K_ss - v.T @ v
    except np.linalg.LinAlgError:
        return np.full(len(x_pred), np.nan), np.full((len(x_pred), len(x_pred)), np.nan)
    return y_mean, y_cov

def predict_marginalized(x_obs, y_obs, cov_obs, x_pred, kernel_func, theta_chain, n_MC=100, **kwargs):
    means, covs = [], []
    theta_indices = np.random.choice(len(theta_chain), n_MC, replace=False)
    
    for idx in theta_indices:
        try:
            mu_s, cov_s = gp_predict(x_obs, y_obs, cov_obs, x_pred, kernel_func, theta_chain[idx], **kwargs)
            means.append(mu_s)
            covs.append(cov_s)
        except np.linalg.LinAlgError:
            continue
    means = np.array(means)
    covs  = np.array(covs)
    mean_final = np.mean(means, axis=0)
    cov_final = np.mean(covs, axis=0) + np.cov(means.T)

    return mean_final, cov_final

#----------

from scipy.special import kv, gamma
from scipy.optimize import minimize

def kernel(x1, x2, p, name):
    d = np.abs(x1[:, None] - x2[None, :])
    if name == 'SE':
        return p[0]**2 * np.exp(-0.5 * (d/p[1])**2)
    elif name == 'DSE':
        return p[0]**2 * np.exp(-0.5 * (d/p[1])**2) + p[2]**2 * np.exp(-0.5 * (d/p[3])**2)
    
    nu_dict = {'Matern32': 1.5, 'Matern52': 2.5, 'Matern72': 3.5, 'Matern92': 4.5}
    nu = nu_dict[name]
    term = np.sqrt(2*nu) * d / p[1]
    term[d==0] = 1e-8 
    K = p[0]**2 * (2**(1-nu)/gamma(nu)) * term**nu * kv(nu, term)
    np.fill_diagonal(K, p[0]**2)
    return K

def neg_log_likelihood(p, x, y, cov, name):
    K = kernel(x, x, p, name) + cov + np.eye(len(x))*1e-8
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return 0.5 * y @ alpha + np.sum(np.log(np.diag(L)))

def train_gp(x, y, cov, name):
    p0 = [np.std(y), (np.max(x)-np.min(x))/2]
    bounds = [(1e-2, 1000), (1e-2, 10)]
    if name == 'DSE':
        p0 += [np.std(y)/2, (np.max(x)-np.min(x))/10]
        bounds += bounds
    res = minimize(neg_log_likelihood, p0, args=(x, y, cov, name), bounds=bounds)
    return res.x

def get_reduced_chi2_samples(x, y, cov, name, n_samples=10000):
    p_best = train_gp(x, y, cov, name)
    
    K = kernel(x, x, p_best, name)
    K_total = K + cov + np.eye(len(x))*1e-8
    L = np.linalg.cholesky(K_total)
    
    # Posterior mean and covariance at observed points
    # mu = K * (K+C)^-1 * y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mu_post = K @ alpha
    
    # cov = K - K * (K+C)^-1 * K
    v = np.linalg.solve(L, K)
    cov_post = K - v.T @ v
    
    # Generate realizations
    realizations = np.random.multivariate_normal(mu_post, cov_post, n_samples)
    
    # Calculate Chi2 for each realization against data
    # Chi2 = (y - y_rec)^T * C^-1 * (y - y_rec)
    inv_cov = np.linalg.inv(cov) # Data covariance inverse
    deltas = y - realizations
    chi2 = np.einsum('ij,jk,ik->i', deltas, inv_cov, deltas)
    
    dof = len(x) - len(p_best)
    return chi2 / dof

# 用于高斯过程回归的通用似然函数
def log_likelihood(theta, x, y, cov, kernel_func, **kwargs):
    l, sigma_f = np.exp(theta)
    K = kernel_func(x, x, [l, sigma_f], **kwargs) 
    A = K + cov + 1e-8 * np.eye(len(x))
    try:
        L = np.linalg.cholesky(A)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        return -0.5 * y @ alpha - np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:
        return -np.inf

def log_probability(theta, x, y, cov, kernel_func, **kwargs):
    log_l, log_sf = theta
    if not (-10 < log_l < 10) or not (-10 < log_sf < 10):
        return -np.inf
    return log_likelihood(theta, x, y, cov, kernel_func, **kwargs)

# def matern_kernel(x1, x2, nu, l, sigma_f):
#     x1 = np.atleast_2d(x1).T if np.ndim(x1) == 1 else np.atleast_2d(x1)
#     x2 = np.atleast_2d(x2)

#     d = np.abs(x1 - x2)

#     if np.isclose(nu, 0.5):
#         K = sigma_f**2 * np.exp(-d / l)

#     elif np.isclose(nu, 1.5):
#         sqrt3 = np.sqrt(3.0)
#         K = sigma_f**2 * (1 + sqrt3 * d / l) * np.exp(-sqrt3 * d / l)

#     elif np.isclose(nu, 2.5):
#         sqrt5 = np.sqrt(5.0)
#         K = sigma_f**2 * (
#             1 + sqrt5 * d / l + 5.0 * d**2 / (3.0 * l**2)
#         ) * np.exp(-sqrt5 * d / l)

#     else:
#         from scipy.special import gamma, kv
#         d_safe = np.maximum(d, 1e-12)
#         arg = np.sqrt(2 * nu) * d_safe / l
#         factor = (2**(1. - nu)) / gamma(nu)
#         K = sigma_f**2 * factor * (arg**nu) * kv(nu, arg)
#         K[d == 0.0] = sigma_f**2

#     return K

# def gp_predict(z_obs, y_obs, cov_obs, z_predict, kernel_func, nu, l, sigma_f):
#     K = kernel_func(z_obs, z_obs, nu, l=l, sigma_f=sigma_f)
#     K_s = kernel_func(z_predict, z_obs, nu, l=l, sigma_f=sigma_f)
#     K_ss = kernel_func(z_predict, z_predict, nu, l=l, sigma_f=sigma_f)

#     A = K + cov_obs
    
#     A += 1e-8 * np.eye(len(A))
    
#     L = np.linalg.cholesky(A)
    
#     alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
#     y_pred_mean = K_s @ alpha
    
#     v = np.linalg.solve(L, K_s.T)
#     y_pred_cov = K_ss - v.T @ v
    
#     return y_pred_mean, y_pred_cov

# def gp_predict_marginalized(z_obs, y_res, cov_obs, z_predict, theta_samples, trend_func, thin, kernel, nu):
#     means = []
#     covs  = []
    
#     z_predict_phys = 10**z_predict # from log space to real
#     trend_at_predict = trend_func(z_predict_phys)

#     for theta in theta_samples[::thin]:
#         l = np.exp(theta[0])
#         sigma_f = np.exp(theta[1])

#         mu_s, cov_s = gp_predict(z_obs, y_res, cov_obs, z_predict, kernel, nu, l=l, sigma_f=sigma_f)
        
#         means.append(mu_s + trend_at_predict)
#         covs.append(cov_s)

#     means = np.array(means)
#     covs  = np.array(covs)

#     mean_final = np.mean(means, axis=0)
#     cov_final = np.mean(covs, axis=0) + np.cov(means.T)

#     return mean_final, cov_final