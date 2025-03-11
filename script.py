import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def calculate_trajectory(v0, theta_deg, h0=0, g=9.8, dt=0.01):
    """
    Calculate the trajectory of a projectile.
    
    Parameters:
    - v0: initial velocity (m/s)
    - theta_deg: launch angle (degrees)
    - h0: initial height (m)
    - g: gravitational acceleration (m/s²)
    - dt: time step for simulation (s)
    
    Returns:
    - x, y: position coordinates arrays
    - t: time array
    """
    # Convert angle to radians
    theta = np.radians(theta_deg)
    
    # Initial velocities
    v0x = v0 * np.cos(theta)
    v0y = v0 * np.sin(theta)
    
    # Calculate time of flight (using quadratic formula)
    # For y(t) = h0 + v0y*t - 0.5*g*t² = 0
    if v0y**2 + 2*g*h0 < 0:  # No real solutions (doesn't reach ground)
        return None, None, None
    
    t_flight = (v0y + np.sqrt(v0y**2 + 2*g*h0)) / g
    
    # Create time array
    t = np.arange(0, t_flight, dt)
    
    # Calculate position at each time step
    x = v0x * t
    y = h0 + v0y * t - 0.5 * g * t**2
    
    # Add the landing point precisely
    t_landing = (v0y + np.sqrt(v0y**2 + 2*g*h0)) / g
    if t[-1] < t_landing:
        t = np.append(t, t_landing)
        x = np.append(x, v0x * t_landing)
        y = np.append(y, 0)  # Landing at y=0
    
    return x, y, t

def calculate_range(v0, theta_deg, h0=0, g=9.8):
    """Calculate the range of a projectile analytically."""
    theta = np.radians(theta_deg)
    if h0 == 0:
        # Simple case: launch and landing at same height
        return (v0**2 * np.sin(2*theta)) / g
    else:
        # Launch from height h0
        return v0 * np.cos(theta) * (v0 * np.sin(theta) + 
                                    np.sqrt((v0 * np.sin(theta))**2 + 2*g*h0)) / g

def plot_trajectories(v0=20, h0=0, g=9.8):
    """Plot multiple trajectories for different launch angles."""
    angles = np.arange(10, 91, 10)  # 10° to 90° in steps of 10°
    plt.figure(figsize=(10, 6))
    
    max_range = 0
    max_height = 0
    
    for theta in angles:
        x, y, _ = calculate_trajectory(v0, theta, h0, g)
        if x is not None and y is not None:
            plt.plot(x, y, label=f'θ = {theta}°')
            max_range = max(max_range, x[-1])
            max_height = max(max_height, np.max(y))
    
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Height (m)')
    plt.title(f'Projectile Trajectories for Different Launch Angles (v₀ = {v0} m/s)')
    plt.legend()
    plt.axis([0, max_range*1.1, 0, max_height*1.1])
    plt.show()

def plot_range_vs_angle(v0=20, h0=0, g=9.8):
    """Plot the range as a function of launch angle."""
    angles = np.linspace(0, 90, 91)  # 0° to 90° in steps of 1°
    ranges = [calculate_range(v0, theta, h0, g) for theta in angles]
    
    plt.figure(figsize=(10, 6))
    plt.plot(angles, ranges)
    plt.grid(True)
    plt.xlabel('Launch Angle (degrees)')
    plt.ylabel('Range (m)')
    plt.title(f'Range vs. Launch Angle (v₀ = {v0} m/s, h₀ = {h0} m)')
    
    # Find and mark the maximum range
    max_range_idx = np.argmax(ranges)
    max_range_angle = angles[max_range_idx]
    max_range_value = ranges[max_range_idx]
    
    plt.scatter(max_range_angle, max_range_value, color='red', s=100, 
                label=f'Maximum Range: {max_range_value:.1f} m at θ = {max_range_angle}°')
    plt.legend()
    plt.show()

def animate_trajectory(v0=20, theta=45, h0=0, g=9.8):
    """Create an animation of the projectile motion."""
    x, y, t = calculate_trajectory(v0, theta, h0, g)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, max(x) * 1.1)
    ax.set_ylim(0, max(y) * 1.1)
    ax.grid(True)
    ax.set_xlabel('Horizontal Distance (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'Projectile Motion (v₀ = {v0} m/s, θ = {theta}°)')
    
    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '-', lw=1, alpha=0.5)
    
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        return line, trace
    
    def animate(i):
        line.set_data([x[i]], [y[i]])
        trace.set_data(x[:i+1], y[:i+1])
        return line, trace
    
    frames = len(x)
    anim = FuncAnimation(fig, animate, frames=frames, 
                         init_func=init, blit=True, interval=50)
    plt.show()
    
    return anim

def compare_initial_velocities():
    """Compare range vs. angle curves for different initial velocities."""
    angles = np.linspace(0, 90, 91)
    velocities = [10, 20, 30, 40]
    
    plt.figure(figsize=(10, 6))
    
    for v0 in velocities:
        ranges = [calculate_range(v0, theta) for theta in angles]
        plt.plot(angles, ranges, label=f'v₀ = {v0} m/s')
    
    plt.grid(True)
    plt.xlabel('Launch Angle (degrees)')
    plt.ylabel('Range (m)')
    plt.title('Range vs. Launch Angle for Different Initial Velocities')
    plt.legend()
    plt.show()

def compare_initial_heights():
    """Compare range vs. angle curves for different initial heights."""
    angles = np.linspace(0, 90, 91)
    heights = [0, 5, 10, 20]
    
    plt.figure(figsize=(10, 6))
    
    for h0 in heights:
        ranges = [calculate_range(20, theta, h0) for theta in angles]
        plt.plot(angles, ranges, label=f'h₀ = {h0} m')
    
    plt.grid(True)
    plt.xlabel('Launch Angle (degrees)')
    plt.ylabel('Range (m)')
    plt.title('Range vs. Launch Angle for Different Initial Heights (v₀ = 20 m/s)')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_trajectories(v0=20)
    plot_range_vs_angle(v0=20)
    compare_initial_velocities()
    compare_initial_heights()
    # To create an animation of a specific trajectory:
    # anim = animate_trajectory(v0=20, theta=45)