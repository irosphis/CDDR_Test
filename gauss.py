import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import kv, gamma
import emcee
import utils
class GPR:import numpy as np
from scipy.optimize import minimize
from scipy.special import kv, gamma
from scipy.linalg import cho_solve, cho_factor
import emcee

class GPR:
    def __init__(self, l=1.0, sigma_f=1.0, kernel_name='rbf', nu=2.5):
        self.l = float(l)
        self.sigma_f = float(sigma_f)
        self.nu = float(nu)
        self.kernel_name = kernel_name
        
        self.X_train = None
        self.y_train = None
        self.noise_cov_matrix = None
        self.L = None 
        self.alpha = None 
        self.hyperparameter_samples = None 

    def _get_kernel_matrix(self, x1, x2):
        if self.kernel_name == 'rbf':
            return self._rbf_kernel(x1, x2)
        elif self.kernel_name == 'matern':
            return self._matern_kernel(x1, x2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")

    def _rbf_kernel(self, x1, x2):
        dist_sq = (x1[:, None] - x2[None, :])**2 
        return self.sigma_f**2 * np.exp(-0.5 * dist_sq / self.l**2)

    def _matern_kernel(self, x1, x2):
        d = np.abs(x1[:, None] - x2[None, :])
        
        if self.nu == 0.5:
            return self.sigma_f**2 * np.exp(-d / self.l)
        elif self.nu == 1.5:
            x = np.sqrt(3) * d / self.l
            return self.sigma_f**2 * (1 + x) * np.exp(-x)
        elif self.nu == 2.5:
            x = np.sqrt(5) * d / self.l
            return self.sigma_f**2 * (1 + x + x**2 / 3) * np.exp(-x)
        
        d[d == 0] = 1e-15 
        sqrt_2nu = np.sqrt(2 * self.nu)
        scaled_d = sqrt_2nu * d / self.l
        coeff = (2**(1 - self.nu)) / gamma(self.nu)
        val = self.sigma_f**2 * coeff * (scaled_d**self.nu) * kv(self.nu, scaled_d)
        
        if x1 is x2: 
            np.fill_diagonal(val, self.sigma_f**2)
            
        return val

    def _log_likelihood(self, theta):
        old_params = (self.l, self.sigma_f)
        self.l, self.sigma_f = np.exp(theta[0]), np.exp(theta[1])
        
        K = self._get_kernel_matrix(self.X_train, self.X_train) + self.noise_cov_matrix
        K_stable = K + 1e-10 * np.eye(len(K))
        L = np.linalg.cholesky(K_stable)
        logdet = 2 * np.sum(np.log(np.diag(L)))
        alpha = cho_solve((L, True), self.y_train)
        term1 = -0.5 * self.y_train.T @ alpha
        
        self.l, self.sigma_f = old_params
        
        return term1 - 0.5 * logdet - 0.5 * len(self.X_train) * np.log(2 * np.pi)

    def fit(self, X_train, y_train, noise_cov_matrix=None, optimize_method='minimize', n_walkers=32, n_steps=2000):
        self.X_train = np.asarray(X_train).ravel()
        self.y_train = np.asarray(y_train).ravel()
        
        if noise_cov_matrix is None:
            self.noise_cov_matrix = 1e-6 * np.eye(len(self.X_train))
        else:
            self.noise_cov_matrix = noise_cov_matrix

        initial_theta = np.array([np.log(self.l), np.log(self.sigma_f)])

        if optimize_method == 'minimize':
            nll = lambda t: -self._log_likelihood(t)
            bounds = ((-10, 10), (-10, 10)) 
            res = minimize(nll, initial_theta, bounds=bounds, method='L-BFGS-B')
            
            self.l, self.sigma_f = np.exp(res.x[0]), np.exp(res.x[1])
            self.hyperparameter_samples = None
            
            K = self._get_kernel_matrix(self.X_train, self.X_train) + self.noise_cov_matrix
            self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))
            self.alpha = cho_solve((self.L, True), self.y_train)

        elif optimize_method == 'emcee':
            n_dim = 2
            pos = initial_theta + 1e-4 * np.random.randn(n_walkers, n_dim)
            
            def log_prob(theta):
                if not (-10 < theta[0] < 10 and -10 < theta[1] < 10):
                    return -np.inf
                return self._log_likelihood(theta)

            sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
            sampler.run_mcmc(pos, n_steps, progress=True)
            
            flat_samples = sampler.get_chain(discard=int(n_steps*0.3), thin=15, flat=True)
            self.hyperparameter_samples = flat_samples
            
            median_theta = np.median(flat_samples, axis=0)
            self.l, self.sigma_f = np.exp(median_theta[0]), np.exp(median_theta[1])
            
            K = self._get_kernel_matrix(self.X_train, self.X_train) + self.noise_cov_matrix
            self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))
            self.alpha = cho_solve((self.L, True), self.y_train)

        else:
            raise ValueError("Invalid optimization method.")

    def predict(self, X_pred, return_cov=True):
        X_pred = np.asarray(X_pred).ravel()
        K_trans = self._get_kernel_matrix(self.X_train, X_pred)
        mu = K_trans.T @ self.alpha
        
        if not return_cov:
            return mu
            
        K_ss = self._get_kernel_matrix(X_pred, X_pred)
        v = cho_solve((self.L, True), K_trans)
        cov = K_ss - K_trans.T @ v
        
        return mu, cov

    def calculate_metrics(self):
        current_theta = np.array([np.log(self.l), np.log(self.sigma_f)])
        lml = self._log_likelihood(current_theta)
        
        mu_pred = self.predict(self.X_train, return_cov=False)
        residuals = self.y_train - mu_pred
        
        inv_noise = np.linalg.inv(self.noise_cov_matrix)
        chi2 = residuals.T @ inv_noise @ residuals
        red_chi2 = chi2 / len(self.X_train)
            
        return lml, red_chi2

    def predict_with_uncertainty(self, X_pred, num_samples=100):
        if self.hyperparameter_samples is None:
            raise RuntimeError("Run fit with optimize_method='emcee' first.")
            
        X_pred = np.asarray(X_pred).ravel()
        indices = np.random.choice(len(self.hyperparameter_samples), num_samples, replace=False)
        
        mus = []
        covs = []
        best_l, best_sigma = self.l, self.sigma_f
        
        for idx in indices:
            theta = self.hyperparameter_samples[idx]
            self.l, self.sigma_f = np.exp(theta[0]), np.exp(theta[1])
            
            K = self._get_kernel_matrix(self.X_train, self.X_train) + self.noise_cov_matrix
            L_sample = np.linalg.cholesky(K + 1e-10*np.eye(len(K)))
            self.L = L_sample
            self.alpha = cho_solve((L_sample, True), self.y_train)
            
            m, c = self.predict(X_pred)
            mus.append(m)
            covs.append(c)
        
        self.l, self.sigma_f = best_l, best_sigma
        K = self._get_kernel_matrix(self.X_train, self.X_train) + self.noise_cov_matrix
        self.L = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))
        self.alpha = cho_solve((self.L, True), self.y_train)
        
        mus = np.array(mus)
        covs = np.array(covs)
        
        total_mu = np.mean(mus, axis=0)
        mean_of_covs = np.mean(covs, axis=0)
        
        centered_mu = mus - total_mu[None, :]
        cov_of_means = (centered_mu.T @ centered_mu) / (len(mus) - 1)
        
        total_cov = mean_of_covs + cov_of_means
        
        return total_mu, total_cov