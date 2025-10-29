import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(1)

# Locations - r_i = 0, 0.01, ..., 0.99  
n_locations = 100
locations = np.linspace(0.0, 0.99, n_locations)

# Parameters 
sigma = 1.0   # variance
rho = 0.5      # length-scale parameter
mu = np.zeros(n_locations)  # Zero means

# Pairwise distance matrix
D = np.abs(locations[:, None] - locations[None, :])

# Covariance kernels
def cov_exp(D, sigma=1.0, rho=0.1):
    return sigma * np.exp(-D / rho)

def cov_exp_squared(D, sigma=1.0, rho=0.1):
    return sigma * np.exp(- (D / rho)**2)

# Build true covariance matrices
Sigma_exp = cov_exp(D, sigma=sigma, rho=rho)
Sigma_sqexp = cov_exp_squared(D, sigma=sigma, rho=rho)

# Sample sizes to simulate
sample_sizes = [100, 1000, 10000]

# Helper to run simulation, estimate, and plot
def simulate_and_plot(Sigma, label):
    print(f"\n=== Kernel: {label} ===")
    for n_samples in sample_sizes:
        # Simulate n_samples independent multivariate normals (n_samples x n_locations)
        samples = np.random.multivariate_normal(mean=mu, cov=Sigma, size=n_samples)
        
        # Sample mean (vector of length n_locations)
        sample_mean = np.mean(samples, axis=0)
        # Sample covariance (n_locations x n_locations)
        sample_cov = np.cov(samples, rowvar=False, bias=False)
        
        print(f"n = {n_samples}: mean (first 5) = {sample_mean[:5]}")
        print(f"n = {n_samples}: top-left 3x3 of sample covariance:\n{sample_cov[:3,:3]}")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        im0 = axes[0].imshow(Sigma, origin='lower', aspect='auto')
        axes[0].set_title('True covariance')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        
        im1 = axes[1].imshow(sample_cov, origin='lower', aspect='auto')
        axes[1].set_title(f'Estimated cov (n={n_samples})')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.suptitle(f'{label} kernel — True vs Estimated covariance')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# Run for exponential kernel
simulate_and_plot(Sigma_exp, 'Exponential (sigma^2 exp(-d/rho))')

# Run for squared-exponential kernel
simulate_and_plot(Sigma_sqexp, 'Squared-exp (sigma^2 exp(-(d/rho)^2))')
