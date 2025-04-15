import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Ensure the images directory exists
img_dir = os.path.join("docs", "1 Physics", "6 Statistics", "images")
os.makedirs(img_dir, exist_ok=True)

class CirclePiEstimator:
    """Estimator for π using the circle method"""
    
    def __init__(self, num_points=10000):
        """
        Initialize the estimator.
        
        Parameters:
        - num_points: number of random points to generate
        """
        self.num_points = num_points
        self.points = np.random.uniform(-1, 1, (num_points, 2))
        self.inside = np.sum(self.points[:, 0]**2 + self.points[:, 1]**2 <= 1)
        self.pi_estimate = 4 * self.inside / num_points
    
    def plot_points(self, filename=None):
        """Plot the random points and circle"""
        plt.figure(figsize=(8, 8))
        
        # Plot points
        inside = self.points[:, 0]**2 + self.points[:, 1]**2 <= 1
        plt.scatter(self.points[inside, 0], self.points[inside, 1], 
                   c='blue', alpha=0.5, label='Inside circle')
        plt.scatter(self.points[~inside, 0], self.points[~inside, 1], 
                   c='red', alpha=0.5, label='Outside circle')
        
        # Plot circle
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        
        # Plot square
        plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=2)
        
        plt.title(f'π Estimation: {self.pi_estimate:.6f} (n={self.num_points})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if filename:
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return plt.gcf()

class BuffonNeedle:
    """Estimator for π using Buffon's Needle method"""
    
    def __init__(self, needle_length=1, line_spacing=2, num_throws=10000):
        """
        Initialize the estimator.
        
        Parameters:
        - needle_length: length of the needle
        - line_spacing: distance between parallel lines
        - num_throws: number of needle throws
        """
        self.needle_length = needle_length
        self.line_spacing = line_spacing
        self.num_throws = num_throws
        
        # Generate random needle positions
        self.x = np.random.uniform(0, line_spacing, num_throws)
        self.theta = np.random.uniform(0, np.pi, num_throws)
        
        # Calculate crossings
        self.crossings = self.x + 0.5 * needle_length * np.sin(self.theta) > line_spacing
        self.crossings += self.x - 0.5 * needle_length * np.sin(self.theta) < 0
        
        self.pi_estimate = (2 * needle_length * num_throws) / (line_spacing * np.sum(self.crossings))
    
    def plot_needles(self, max_needles=100, filename=None):
        """Plot the needle positions"""
        plt.figure(figsize=(10, 6))
        
        # Plot lines
        for i in range(0, int(self.line_spacing * 2) + 1):
            plt.axhline(y=i, color='k', linestyle='-', alpha=0.5)
        
        # Plot needles
        n = min(max_needles, self.num_throws)
        for i in range(n):
            x1 = self.x[i] - 0.5 * self.needle_length * np.cos(self.theta[i])
            x2 = self.x[i] + 0.5 * self.needle_length * np.cos(self.theta[i])
            y1 = i % self.line_spacing
            y2 = y1 + self.needle_length * np.sin(self.theta[i])
            
            color = 'red' if self.crossings[i] else 'blue'
            plt.plot([x1, x2], [y1, y2], color=color, alpha=0.5)
        
        plt.title(f'Buffon\'s Needle: π ≈ {self.pi_estimate:.6f} (n={self.num_throws})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        
        if filename:
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        
        return plt.gcf()

def run_circle_simulation(num_points_list):
    """Run simulation for different numbers of points"""
    results = []
    for n in tqdm(num_points_list, desc="Running circle simulation"):
        estimator = CirclePiEstimator(n)
        results.append({
            'n': n,
            'pi_estimate': estimator.pi_estimate,
            'error': abs(estimator.pi_estimate - np.pi)
        })
        if n in [100, 1000, 10000, 100000]:
            estimator.plot_points(f'circle_pi_n{n}.png')
    return results

def run_buffon_simulation(num_throws_list):
    """Run simulation for different numbers of throws"""
    results = []
    for n in tqdm(num_throws_list, desc="Running Buffon's Needle simulation"):
        estimator = BuffonNeedle(num_throws=n)
        results.append({
            'n': n,
            'pi_estimate': estimator.pi_estimate,
            'error': abs(estimator.pi_estimate - np.pi)
        })
        if n in [100, 1000, 10000, 100000]:
            estimator.plot_needles(filename=f'buffon_pi_n{n}.png')
    return results

def compare_methods():
    """Compare both methods"""
    num_points = [100, 1000, 10000, 100000, 1000000]
    
    # Run both simulations
    circle_results = run_circle_simulation(num_points)
    buffon_results = run_buffon_simulation(num_points)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot([r['n'] for r in circle_results], 
             [r['error'] for r in circle_results], 
             'b-', label='Circle Method')
    plt.plot([r['n'] for r in buffon_results], 
             [r['error'] for r in buffon_results], 
             'r-', label='Buffon\'s Needle')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of points/throws')
    plt.ylabel('Absolute error')
    plt.title('Comparison of π Estimation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(img_dir, 'pi_estimation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return circle_results, buffon_results

if __name__ == "__main__":
    print("Running π estimation simulations...")
    circle_results, buffon_results = compare_methods()
    print("\nSimulation complete! Images saved to:", img_dir)
    
    # Print final results
    print("\nFinal Results:")
    print("Circle Method:")
    for r in circle_results:
        print(f"n={r['n']}: π ≈ {r['pi_estimate']:.6f} (error: {r['error']:.6f})")
    
    print("\nBuffon's Needle:")
    for r in buffon_results:
        print(f"n={r['n']}: π ≈ {r['pi_estimate']:.6f} (error: {r['error']:.6f})") 