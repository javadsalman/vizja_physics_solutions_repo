import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
from scipy.optimize import curve_fit
from IPython.display import HTML

# Create images directory if it doesn't exist
img_dir = os.path.join("docs", "1 Physics", "7 Measurements", "images")
os.makedirs(img_dir, exist_ok=True)

# Experimental data
trials = np.arange(1, 11)
times_10_oscillations = np.array([20.64, 20.58, 20.71, 20.62, 20.59, 20.67, 20.60, 20.65, 20.69, 20.63])
length = 1.053  # meters
length_uncertainty = 0.0005  # meters

# Calculate statistics
mean_time_10 = np.mean(times_10_oscillations)
std_dev = np.std(times_10_oscillations, ddof=1)
uncertainty_mean_10 = std_dev / np.sqrt(len(times_10_oscillations))

# Calculate period and uncertainty
period = mean_time_10 / 10
period_uncertainty = uncertainty_mean_10 / 10

# Calculate g and uncertainty
g_measured = 4 * np.pi**2 * length / period**2
g_uncertainty = g_measured * np.sqrt((length_uncertainty/length)**2 + (2*period_uncertainty/period)**2)

# Theoretical pendulum period function
def pendulum_period(L, g):
    return 2 * np.pi * np.sqrt(L / g)

# Function for pendulum position at time t
def pendulum_position(t, L, g, theta_max=10):
    T = pendulum_period(L, g)
    theta = theta_max * np.cos(2*np.pi*t/T)
    x = L * np.sin(np.radians(theta))
    y = -L * np.cos(np.radians(theta))
    return x, y

# Create data visualization functions
def plot_measurement_histogram():
    plt.figure(figsize=(10, 6))
    
    # Create histogram of measurements
    plt.hist(times_10_oscillations, bins=5, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add mean line
    plt.axvline(mean_time_10, color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {mean_time_10:.3f} s')
    
    # Add standard deviation range
    plt.axvline(mean_time_10 - std_dev, color='green', linestyle='dotted', linewidth=2)
    plt.axvline(mean_time_10 + std_dev, color='green', linestyle='dotted', linewidth=2,
                label=f'Std Dev: {std_dev:.3f} s')
    
    plt.title('Distribution of Time Measurements for 10 Oscillations')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(img_dir, 'time_distribution.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_pendulum_animation():
    # Set up the figure and axis for the animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 0.2)
    ax.set_aspect('equal')
    
    # Create pendulum elements
    line, = ax.plot([], [], 'k-', lw=2)
    mass, = ax.plot([], [], 'ro', markersize=12)
    time_text = ax.text(0.02, 0.02, '', transform=ax.transAxes)
    
    # Initialize function for animation
    def init():
        line.set_data([], [])
        mass.set_data([], [])
        time_text.set_text('')
        return line, mass, time_text
    
    # Animation update function
    def update(frame):
        t = frame / 50  # time in seconds
        x, y = pendulum_position(t, length, g_measured)
        
        line.set_data([0, x], [0, y])
        mass.set_data(x, y)
        
        period_val = pendulum_period(length, g_measured)
        time_text.set_text(f'Time: {t:.2f} s\nPeriod: {period_val:.3f} s')
        
        return line, mass, time_text
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)
    
    # Save as GIF
    save_path = os.path.join(img_dir, 'pendulum_animation.gif')
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()
    
    return save_path

def plot_error_analysis():
    plt.figure(figsize=(12, 6))
    
    # Create bar graph of relative contributions to uncertainty
    relative_error_L = (length_uncertainty/length)**2 / ((length_uncertainty/length)**2 + (2*period_uncertainty/period)**2) * 100
    relative_error_T = (2*period_uncertainty/period)**2 / ((length_uncertainty/length)**2 + (2*period_uncertainty/period)**2) * 100
    
    labels = ['Length Measurement', 'Time Measurement']
    values = [relative_error_L, relative_error_T]
    
    plt.bar(labels, values, color=['blue', 'orange'])
    plt.title('Relative Contributions to Uncertainty in g Measurement')
    plt.ylabel('Contribution to Total Uncertainty (%)')
    plt.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    save_path = os.path.join(img_dir, 'error_contributions.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_g_comparison():
    plt.figure(figsize=(10, 6))
    
    # Standard value
    g_standard = 9.81
    
    # Plot measured value with error bars
    plt.errorbar(x=['Measured'], y=[g_measured], yerr=[g_uncertainty], 
                 fmt='o', color='blue', ecolor='blue', capsize=10, markersize=8)
    
    # Plot standard value
    plt.scatter(['Standard'], [g_standard], color='red', marker='s', s=100, label='Standard Value')
    
    plt.title('Comparison of Measured Gravitational Acceleration with Standard Value')
    plt.ylabel('g (m/s²)')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal lines
    plt.axhline(y=g_measured, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=g_standard, color='red', linestyle='--', alpha=0.5)
    
    # Add text annotations
    plt.text(1.1, g_measured, f"Measured: {g_measured:.4f} ± {g_uncertainty:.4f} m/s²", va='center')
    plt.text(1.1, g_standard, f"Standard: {g_standard:.4f} m/s²", va='center')
    
    save_path = os.path.join(img_dir, 'g_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_period_vs_length():
    # Generate data for various pendulum lengths
    lengths = np.linspace(0.5, 2.0, 100)
    
    # Theoretical periods for g = 9.81
    periods_theory = pendulum_period(lengths, 9.81)
    
    # Measured periods based on our g value
    periods_measured = pendulum_period(lengths, g_measured)
    
    plt.figure(figsize=(10, 6))
    
    # Plot theoretical curve
    plt.plot(lengths, periods_theory, 'r-', label=f'Theoretical (g = 9.81 m/s²)')
    
    # Plot measured curve
    plt.plot(lengths, periods_measured, 'b--', label=f'Our measurement (g = {g_measured:.3f} m/s²)')
    
    # Plot our actual data point
    plt.scatter([length], [period], color='green', s=100, 
                label=f'Our experiment: L={length}m, T={period:.3f}s')
    
    plt.title('Pendulum Period vs. Length')
    plt.xlabel('Length (m)')
    plt.ylabel('Period (s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_path = os.path.join(img_dir, 'period_vs_length.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

# Generate all plots and animations
def generate_all_visualizations():
    """Generate all visualizations for the pendulum experiment"""
    results = {
        'histogram': plot_measurement_histogram(),
        'error_analysis': plot_error_analysis(),
        'g_comparison': plot_g_comparison(),
        'period_vs_length': plot_period_vs_length()
    }
    
    try:
        results['animation'] = plot_pendulum_animation()
    except Exception as e:
        print(f"Animation could not be generated: {e}")
        
    return results

if __name__ == "__main__":
    # Print basic results
    print(f"Mean time for 10 oscillations: {mean_time_10:.3f} ± {uncertainty_mean_10:.3f} s")
    print(f"Period: {period:.4f} ± {period_uncertainty:.4f} s")
    print(f"Measured g: {g_measured:.4f} ± {g_uncertainty:.4f} m/s²")
    print(f"Standard g: 9.8100 m/s²")
    print(f"Difference: {abs(g_measured - 9.81):.4f} m/s²")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_paths = generate_all_visualizations()
    
    # Report where images were saved
    print(f"\nImages saved to: {img_dir}")
    for name, path in viz_paths.items():
        print(f"- {name}: {path}") 