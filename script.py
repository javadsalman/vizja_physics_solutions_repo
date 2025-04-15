import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Ensure the images directory exists
img_dir = os.path.join("docs", "1 Physics", "6 Statistics", "images")
os.makedirs(img_dir, exist_ok=True)

class CLTSimulator:
    """Simulator for demonstrating the Central Limit Theorem"""
    
    def __init__(self, population_size=10000, num_samples=1000):
        """
        Initialize the simulator.
        
        Parameters:
        - population_size: size of the population to generate
        - num_samples: number of samples to draw for each sample size
        """
        self.population_size = population_size
        self.num_samples = num_samples
        self.sample_sizes = [5, 10, 30, 50, 100]  # Different sample sizes to test
        
    def generate_population(self, dist_type, **params):
        """
        Generate a population from a specified distribution.
        
        Parameters:
        - dist_type: type of distribution ('uniform', 'exponential', 'binomial')
        - params: parameters for the distribution
        
        Returns:
        - population array
        """
        if dist_type == 'uniform':
            return np.random.uniform(params.get('low', 0), 
                                   params.get('high', 1), 
                                   self.population_size)
        elif dist_type == 'exponential':
            return np.random.exponential(params.get('scale', 1), 
                                       self.population_size)
        elif dist_type == 'binomial':
            return np.random.binomial(params.get('n', 10), 
                                    params.get('p', 0.5), 
                                    self.population_size)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    def simulate_sampling_distribution(self, population, sample_size):
        """
        Simulate the sampling distribution of the mean for a given sample size.
        
        Parameters:
        - population: the population array
        - sample_size: size of each sample
        
        Returns:
        - array of sample means
        """
        sample_means = np.zeros(self.num_samples)
        for i in range(self.num_samples):
            sample = np.random.choice(population, size=sample_size, replace=True)
            sample_means[i] = np.mean(sample)
        return sample_means
    
    def plot_distribution(self, data, title, filename=None):
        """
        Plot a distribution with its theoretical normal approximation.
        
        Parameters:
        - data: array of sample means
        - title: plot title
        - filename: if provided, save to this filename
        """
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        sns.histplot(data, kde=True, stat='density', 
                    color='skyblue', edgecolor='black')
        
        # Plot normal approximation
        x = np.linspace(min(data), max(data), 100)
        mu, std = np.mean(data), np.std(data)
        y = stats.norm.pdf(x, mu, std)
        plt.plot(x, y, 'r-', linewidth=2, label='Normal Approximation')
        
        plt.title(title)
        plt.xlabel('Sample Mean')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return plt.gcf()
    
    def run_simulation(self, dist_type, **params):
        """
        Run a complete simulation for a given distribution.
        
        Parameters:
        - dist_type: type of distribution
        - params: parameters for the distribution
        
        Returns:
        - dictionary of results
        """
        print(f"\nRunning simulation for {dist_type} distribution")
        
        # Generate population
        population = self.generate_population(dist_type, **params)
        pop_mean = np.mean(population)
        pop_std = np.std(population)
        
        print(f"Population mean: {pop_mean:.4f}")
        print(f"Population standard deviation: {pop_std:.4f}")
        
        results = {
            'population': population,
            'sample_means': {},
            'plots': {}
        }
        
        # Simulate for each sample size
        for n in self.sample_sizes:
            print(f"  Sample size: {n}")
            sample_means = self.simulate_sampling_distribution(population, n)
            results['sample_means'][n] = sample_means
            
            # Calculate theoretical parameters
            theo_mean = pop_mean
            theo_std = pop_std / np.sqrt(n)
            
            # Plot results
            title = f"Sampling Distribution (n={n})\n" + \
                   f"Theoretical: N({theo_mean:.2f}, {theo_std**2:.4f})"
            filename = f"{dist_type}_n{n}.png"
            plot_path = self.plot_distribution(sample_means, title, filename)
            results['plots'][n] = plot_path
            
            # Print statistics
            print(f"    Sample mean: {np.mean(sample_means):.4f}")
            print(f"    Sample std: {np.std(sample_means):.4f}")
            print(f"    Theoretical std: {theo_std:.4f}")
        
        return results

def main():
    """Run simulations for different distributions"""
    simulator = CLTSimulator()
    
    # Run simulations for different distributions
    distributions = {
        'uniform': {'low': 0, 'high': 1},
        'exponential': {'scale': 1},
        'binomial': {'n': 10, 'p': 0.5}
    }
    
    results = {}
    for dist_type, params in distributions.items():
        results[dist_type] = simulator.run_simulation(dist_type, **params)
    
    return results

if __name__ == "__main__":
    main()