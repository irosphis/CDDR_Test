import os
import numpy as np
import numpy as np
from scipy.interpolate import interp1d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import chi2

c_light = 299792.458  # km/s


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
    gp = GaussianProcessRegressor( kernel=kernel, alpha=DL_err**2, n_restarts_optimizer=10)
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
    """
    Load BAO datasets including:
    - DV/rd measurements
    - DM/rd measurements
    - Reconstructed H(z)

    Parameters
    ----------
    project_root : str or None
        The absolute path to the project root.  
        If None, project_root is automatically set to the parent directory 
        of the current working directory (useful when running inside notebooks).

    Returns
    -------
    dict
        {
            "DV": {
                "z": array,
                "value": array,
                "error": array
            },
            "DM": {
                "z": array,
                "value": array,
                "error": array
            },
            "Hz": {
                "z": array (possibly None if not included),
                "value": array,
                "error": array
            }
        }
    """

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
