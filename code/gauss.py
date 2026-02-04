import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import kv, gamma
import emcee
import utils
class GPR:
    def __init__(self, l=1.0, sigma_f=1.0, kernel_name='rbf', nu=1.5):
        self.l, self.sigma_f, self.nu = l, sigma_f, nu
        self.X_train, self.y_train, self.noise_cov_matrix = None, None, None
        self.set_kernel(kernel_name)
        self.hyperparameter_samples = None # Store MCMC samples here

    def set_kernel(self, kernel_name):
        if kernel_name == 'rbf': self._current_kernel = self._rbf_kernel
        elif kernel_name == 'matern': self._current_kernel = self._matern_kernel
        else: raise ValueError(f"Unknown kernel: {kernel_name}")

    def _rbf_kernel(self, x1, x2):
        dist_sq = (x1[:, None] - x2[None, :])**2 
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * dist_sq)

    def _matern_kernel(self, x1, x2):
        r = np.sqrt((x1[:, None] - x2[None, :])**2)
        if self.nu == 0.5:
            return self.sigma_f**2 * np.exp(-r / self.l)
        elif self.nu == 1.5:
            scaled_r = np.sqrt(3) * r / self.l
            return self.sigma_f**2 * (1 + scaled_r) * np.exp(-scaled_r)
        elif self.nu == 2.5:
            scaled_r = np.sqrt(5) * r / self.l
            return self.sigma_f**2 * (1 + scaled_r + scaled_r**2 / 3) * np.exp(-scaled_r)
        else:
            raise NotImplementedError(f"Matern kernel with nu={self.nu} not explicitly implemented for simplicity.")

    def _log_likelihood(self, theta):
        original_l, original_sigma_f, original_nu = self.l, self.sigma_f, self.nu
        self.l, self.sigma_f = np.exp(theta[0]), np.exp(theta[1])
        K = self._current_kernel(self.X_train, self.X_train) + self.noise_cov_matrix
        self.l, self.sigma_f, self.nu = original_l, original_sigma_f, original_nu

        sign, logdet = np.linalg.slogdet(K)
        if sign == 0: return -np.inf
        
        return -0.5 * self.y_train.T @ utils.inv(K) @ self.y_train - 0.5 * logdet - 0.5 * len(self.X_train) * np.log(2 * np.pi)

    def _log_prior(self, theta):
        l, sigma_f = np.exp(theta[0]), np.exp(theta[1])
        # Example flat prior for l and sigma_f (adjust bounds as needed)
        if 0.0001 < l < 100 and 0.01 < sigma_f < 1000:
            return 0.0
        return -np.inf

    def _log_posterior(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp): return -np.inf
        return lp + self._log_likelihood(theta)

    def fit(self, X_train, y_train, noise_cov_matrix=None, optimize_method='minimize', n_walkers=32, n_steps=2000):
        self.X_train, self.y_train = np.asarray(X_train).ravel(), np.asarray(y_train).ravel()
        self.noise_cov_matrix = noise_cov_matrix if noise_cov_matrix is not None else 1e-8 * np.eye(len(X_train))

        if optimize_method == 'minimize':
            initial_theta = np.array([np.log(self.l), np.log(self.sigma_f)])
            res = minimize(lambda t: -self._log_likelihood(t), initial_theta, bounds=((None, None), (None, None)), method='L-BFGS-B')
            self.l, self.sigma_f = np.exp(res.x[0]), np.exp(res.x[1])
            self.hyperparameter_samples = None # Clear samples if not MCMC
        elif optimize_method == 'emcee':
            n_dim = 2
            initial_pos = np.array([np.log(self.l), np.log(self.sigma_f)]) + 1e-4 * np.random.randn(n_walkers, n_dim)
            
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, self._log_posterior)
            sampler.run_mcmc(initial_pos, n_steps, progress=True)
            
            # Save the full chain of samples
            self.hyperparameter_samples = sampler.get_chain(discard=int(n_steps * 0.3), thin=15, flat=True)
            
            # For predict() method, set hyperparameters to median of posterior
            best_theta = np.median(self.hyperparameter_samples, axis=0) 
            self.l, self.sigma_f = np.exp(best_theta[0]), np.exp(best_theta[1])
        else:
            raise ValueError(f"Unknown optimization method: {optimize_method}. Choose 'minimize' or 'emcee'.")

    def _predict_single_theta(self, X_pred, l_val, sigma_f_val, nu_val):
        """Helper to predict with a single set of hyperparameters."""
        original_l, original_sigma_f, original_nu = self.l, self.sigma_f, self.nu
        self.l, self.sigma_f, self.nu = l_val, sigma_f_val, nu_val # Temporarily set
        
        K_train_train = self._current_kernel(self.X_train, self.X_train) + self.noise_cov_matrix
        K_pred_pred = self._current_kernel(X_pred, X_pred)
        K_train_pred = self._current_kernel(self.X_train, X_pred)
        
        # K_train_train_inv = np.linalg.inv(K_train_train)
        K_train_train_inv = utils.inv(K_train_train)
        
        mu_pred = K_train_pred.T @ K_train_train_inv @ self.y_train
        cov_pred = K_pred_pred - K_train_pred.T @ K_train_train_inv @ K_train_pred
        
        self.l, self.sigma_f, self.nu = original_l, original_sigma_f, original_nu # Restore
        return mu_pred, cov_pred # Return diagonal of covariance (variance)

    def predict(self, X_pred):
        """
        Predicts mean and covariance for new input points using point-estimated hyperparameters.
        X_pred: Input points for prediction (1D numpy array).
        """
        return self._predict_single_theta(X_pred, self.l, self.sigma_f, self.nu)

    def predict_with_uncertainty(self, X_pred, num_samples=100):
        """
        Predicts mean and variance for new input points, propagating hyperparameter uncertainty.
        X_pred: Input points for prediction (1D numpy array).
        num_samples: Number of hyperparameter samples to use from the posterior.
        """
        if self.hyperparameter_samples is None:
            raise RuntimeError("Hyperparameter samples are not available. Run fit with optimize_method='emcee' first.")
        
        X_pred = np.asarray(X_pred).ravel()
        num_prediction_points = len(X_pred)
        
        # Initialize arrays to store predictions for each hyperparameter sample
        all_mu_preds = np.zeros((num_samples, num_prediction_points))
        all_cov_preds = np.zeros((num_samples, num_prediction_points, num_prediction_points)) # Store full covariance

        sample_indices = np.random.choice(len(self.hyperparameter_samples), num_samples, replace=False)
        
        for i, idx in enumerate(sample_indices):
            log_l, log_sigma_f = self.hyperparameter_samples[idx]
            l_sample, sigma_f_sample = np.exp(log_l), np.exp(log_sigma_f)
            
            mu_sample, cov_sample = self._predict_single_theta(X_pred, l_sample, sigma_f_sample, self.nu)
            all_mu_preds[i] = mu_sample
            all_cov_preds[i] = cov_sample
            
        # # Combine predictions
        # # Total mean is the average of means from each sample
        # mean_of_mus = np.mean(all_mu_preds, axis=0)
        
        # # Total covariance is E[Cov(f|X,y,theta)] + Cov(E[f|X,y,theta])
        # # E[Cov(f|X,y,theta)] is the average of covariance matrices from each sample
        # avg_cov_from_samples = np.mean(all_cov_preds, axis=0)
        
        # # Cov(E[f|X,y,theta]) is the covariance of the means from each sample
        # # This requires calculating the outer product of (mu_sample - mean_of_mus) for each sample
        # var_of_mus_from_samples = np.cov(all_mu_preds.T) # .T because np.cov expects rows as variables
        
        # # Total covariance incorporating hyperparameter uncertainty
        # total_cov_pred = avg_cov_from_samples + var_of_mus_from_samples

        mean_of_mus = np.mean(all_mu_preds, axis=0)

        centered = all_mu_preds - mean_of_mus[None, :]
        var_of_mus_from_samples = centered.T @ centered / (num_samples - 1)

        avg_cov_from_samples = np.mean(all_cov_preds, axis=0)

        total_cov_pred = avg_cov_from_samples + var_of_mus_from_samples

        # Symmetrize
        total_cov_pred = 0.5 * (total_cov_pred + total_cov_pred.T)

        # PSD projection
        eigvals, eigvecs = np.linalg.eigh(total_cov_pred)
        eigvals = np.clip(eigvals, 0, None)
        total_cov_pred = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Tiny jitter
        total_cov_pred += np.eye(len(total_cov_pred)) * 1e-12


        return mean_of_mus, total_cov_pred