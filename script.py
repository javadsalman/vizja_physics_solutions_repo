import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def pendulum_ode(t, y, b, g, L, A, omega):
    """
    Define the ODE system for a forced damped pendulum
    y[0] = theta, y[1] = dtheta/dt
    """
    return [
        y[1],
        -b * y[1] - (g/L) * np.sin(y[0]) + A * np.cos(omega * t)
    ]

def simulate_pendulum(tspan, y0, b, g, L, A, omega):
    """
    Simulate the pendulum motion over a time span
    """
    sol = solve_ivp(
        lambda t, y: pendulum_ode(t, y, b, g, L, A, omega),
        [tspan[0], tspan[-1]],
        y0,
        t_eval=tspan,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    return sol.t, sol.y[0], sol.y[1]

def plot_time_series(t, theta, omega, title="Pendulum Motion"):
    """
    Plot the time series of the pendulum angle
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, theta)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'pendulum_timeseries_omega_{omega:.2f}.png', dpi=300)
    plt.show()

def plot_phase_portrait(theta, omega_values, title="Phase Portrait"):
    """
    Plot the phase portrait (theta vs. dtheta/dt)
    """
    plt.figure(figsize=(10, 8))
    for i, omega in enumerate(omega_values):
        plt.plot(theta[i], omega[i], label=f'ω = {omega:.2f} rad/s')
    
    plt.xlabel('θ (rad)')
    plt.ylabel('dθ/dt (rad/s)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig('pendulum_phase_portrait.png', dpi=300)
    plt.show()

def create_poincare_section(t, theta, dtheta, omega, driving_period):
    """
    Create a Poincaré section by sampling the phase space 
    at times that are multiples of the driving period
    """
    # Find indices where time is approximately a multiple of the driving period
    indices = []
    period = 2 * np.pi / omega
    for i in range(len(t)):
        if abs(t[i] % period) < 1e-10 or abs(t[i] % period - period) < 1e-10:
            indices.append(i)
    
    return theta[indices], dtheta[indices]

def plot_poincare_section(theta_values, dtheta_values, omega_values):
    """
    Plot Poincaré sections for different parameter values
    """
    plt.figure(figsize=(12, 10))
    for i, omega in enumerate(omega_values):
        plt.scatter(theta_values[i], dtheta_values[i], s=5, 
                    label=f'ω = {omega:.2f} rad/s')
    
    plt.xlabel('θ (rad)')
    plt.ylabel('dθ/dt (rad/s)')
    plt.title('Poincaré Section')
    plt.grid(True)
    plt.legend()
    plt.savefig('pendulum_poincare_section.png', dpi=300)
    plt.show()

def create_bifurcation_diagram(A_values, omega, b, g, L):
    """
    Create a bifurcation diagram by varying the driving amplitude
    """
    theta_values = []
    
    # Time settings for simulation
    tmax = 200  # Simulate for a long time to reach steady state
    transient = 100  # Discard the first transient seconds
    t = np.linspace(0, tmax, 10000)
    
    for A in A_values:
        # Simulate with current parameter values
        _, theta, _ = simulate_pendulum(t, [0.1, 0], b, g, L, A, omega)
        
        # Find indices for the steady state (after transient)
        steady_idx = t > transient
        t_steady = t[steady_idx]
        theta_steady = theta[steady_idx]
        
        # Sample at the driving period (stroboscopic sampling)
        driving_period = 2 * np.pi / omega
        sample_indices = []
        
        for i in range(len(t_steady)):
            if abs((t_steady[i] % driving_period) - driving_period) < 1e-2 or abs(t_steady[i] % driving_period) < 1e-2:
                sample_indices.append(i)
        
        # Append sampled theta values to the list
        theta_values.append(theta_steady[sample_indices])
    
    return A_values, theta_values

def plot_bifurcation_diagram(A_values, theta_values):
    """
    Plot the bifurcation diagram
    """
    plt.figure(figsize=(12, 8))
    
    for i, A in enumerate(A_values):
        # Create vertical scatter plot for each A value
        y = theta_values[i]
        x = np.full_like(y, A)
        plt.scatter(x, y, s=0.5, c='black', alpha=0.5)
    
    plt.xlabel('Driving Amplitude (A)')
    plt.ylabel('θ (rad)')
    plt.title('Bifurcation Diagram')
    plt.grid(True)
    plt.savefig('pendulum_bifurcation_diagram.png', dpi=300)
    plt.show()

def animate_pendulum(t, theta, L=1.0, fps=30):
    """
    Create an animation of the pendulum motion
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5*L, 1.5*L)
    ax.set_ylim(-1.5*L, 1.5*L)
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Initialize the pendulum components
    line, = ax.plot([], [], 'k-', lw=2)  # pendulum rod
    mass, = ax.plot([], [], 'bo', markersize=15)  # pendulum mass
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes)
    
    def init():
        line.set_data([], [])
        mass.set_data([], [])
        time_text.set_text('')
        return line, mass, time_text
    
    def update(frame):
        i = frame
        if i < len(t):
            x = L * np.sin(theta[i])
            y = -L * np.cos(theta[i])
            
            line.set_data([0, x], [0, y])
            mass.set_data([x], [y])
            time_text.set_text(f'Time: {t[i]:.2f} s')
        
        return line, mass, time_text
    
    # Create animation
    num_frames = min(len(t), int(t[-1] * fps))
    frame_indices = np.linspace(0, len(t)-1, num_frames, dtype=int)
    
    anim = FuncAnimation(fig, update, frames=frame_indices, init_func=init, blit=True, interval=1000/fps)
    
    # Save animation
    anim.save('pendulum_animation.mp4', writer='ffmpeg', fps=fps, dpi=200)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Parameters
    g = 9.81  # acceleration due to gravity (m/s^2)
    L = 1.0   # pendulum length (m)
    b = 0.2   # damping coefficient
    
    # Time settings
    t_max = 60  # maximum simulation time (s)
    dt = 0.01   # time step (s)
    t = np.arange(0, t_max, dt)
    
    # Initial conditions
    y0 = [np.pi/4, 0]  # [theta_0, omega_0]
    
    # Simulation for various driving parameters
    omega_values = [0.5, 1.0, 2.0, 3.0]  # driving frequencies (rad/s)
    A = 1.0  # driving amplitude
    
    # Store results
    theta_results = []
    dtheta_results = []
    poincare_theta = []
    poincare_dtheta = []
    
    # Run simulations for different frequencies
    for omega in omega_values:
        t_sim, theta, dtheta = simulate_pendulum(t, y0, b, g, L, A, omega)
        theta_results.append(theta)
        dtheta_results.append(dtheta)
        
        # Create Poincaré section
        theta_p, dtheta_p = create_poincare_section(t_sim, theta, dtheta, omega, 2*np.pi/omega)
        poincare_theta.append(theta_p)
        poincare_dtheta.append(dtheta_p)
        
        # Plot time series for the current frequency
        plot_time_series(t_sim, theta, omega, f"Pendulum Motion (ω = {omega:.2f} rad/s)")
    
    # Plot phase portrait for all frequencies
    plot_phase_portrait(theta_results, dtheta_results, "Phase Portrait for Different Driving Frequencies")
    
    # Plot Poincaré sections
    plot_poincare_section(poincare_theta, poincare_dtheta, omega_values)
    
    # Create and plot bifurcation diagram
    A_values = np.linspace(0.1, 2.0, 100)
    omega_fixed = 2.0  # Fixed driving frequency for bifurcation analysis
    A_vals, theta_vals = create_bifurcation_diagram(A_values, omega_fixed, b, g, L)
    plot_bifurcation_diagram(A_vals, theta_vals)
    
    # Create animation for one specific case
    animate_pendulum(t_sim, theta_results[2], L=L)