import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# Ensure the images directory exists
img_dir = os.path.join("docs", "1 Physics", "4 Electromagnetism", "images")
os.makedirs(img_dir, exist_ok=True)

class LorentzForceSimulator:
    """Simulator for charged particle motion under Lorentz force"""
    
    def __init__(self, q=1.0, m=1.0, dt=0.01, tmax=10.0):
        """
        Initialize the simulator.
        
        Parameters:
        - q: charge of the particle (C)
        - m: mass of the particle (kg)
        - dt: time step for simulation (s)
        - tmax: maximum simulation time (s)
        """
        self.q = q  # charge
        self.m = m  # mass
        self.dt = dt  # time step
        self.tmax = tmax  # maximum simulation time
        self.t = np.arange(0, tmax, dt)  # time array
        self.num_steps = len(self.t)
        
        # Initialize arrays for position and velocity
        self.r = np.zeros((self.num_steps, 3))  # position: [x, y, z]
        self.v = np.zeros((self.num_steps, 3))  # velocity: [vx, vy, vz]
        
        # Field configuration (to be set by specific methods)
        self.E = np.zeros(3)  # Electric field vector
        self.B = np.zeros(3)  # Magnetic field vector
        self.field_type = "None"  # Description of field configuration
        
    def set_initial_conditions(self, r0, v0):
        """
        Set initial position and velocity.
        
        Parameters:
        - r0: initial position [x0, y0, z0] (m)
        - v0: initial velocity [vx0, vy0, vz0] (m/s)
        """
        self.r[0] = np.array(r0)
        self.v[0] = np.array(v0)
    
    def set_uniform_B_field(self, B):
        """Set a uniform magnetic field."""
        self.B = np.array(B)
        self.E = np.zeros(3)
        self.field_type = "Uniform B"
        
    def set_uniform_E_field(self, E):
        """Set a uniform electric field."""
        self.E = np.array(E)
        self.B = np.zeros(3)
        self.field_type = "Uniform E"
        
    def set_uniform_EB_fields(self, E, B):
        """Set uniform electric and magnetic fields."""
        self.E = np.array(E)
        self.B = np.array(B)
        self.field_type = "Uniform E and B"
    
    def lorentz_force(self, r, v):
        """
        Calculate acceleration due to Lorentz force.
        
        Parameters:
        - r: position vector [x, y, z]
        - v: velocity vector [vx, vy, vz]
        
        Returns:
        - acceleration vector [ax, ay, az]
        """
        F = self.q * (self.E + np.cross(v, self.B))
        a = F / self.m
        return a
    
    def rk4_step(self, r, v, dt):
        """
        Perform one step of 4th-order Runge-Kutta integration.
        
        Parameters:
        - r: current position
        - v: current velocity
        - dt: time step
        
        Returns:
        - new position, new velocity
        """
        # Stage 1
        a1 = self.lorentz_force(r, v)
        k1r = v
        k1v = a1
        
        # Stage 2
        a2 = self.lorentz_force(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
        k2r = v + 0.5*dt*k1v
        k2v = a2
        
        # Stage 3
        a3 = self.lorentz_force(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
        k3r = v + 0.5*dt*k2v
        k3v = a3
        
        # Stage 4
        a4 = self.lorentz_force(r + dt*k3r, v + dt*k3v)
        k4r = v + dt*k3v
        k4v = a4
        
        # Combine stages
        r_new = r + (dt/6)*(k1r + 2*k2r + 2*k3r + k4r)
        v_new = v + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
        
        return r_new, v_new
    
    def run_simulation(self):
        """Execute the simulation using RK4 integration."""
        for i in range(1, self.num_steps):
            self.r[i], self.v[i] = self.rk4_step(self.r[i-1], self.v[i-1], self.dt)
    
    def calculate_parameters(self):
        """Calculate relevant physical parameters based on the simulation."""
        results = {
            "charge": self.q,
            "mass": self.m,
            "fields": {
                "type": self.field_type,
                "E": self.E,
                "B": self.B
            }
        }
        
        # Calculate Larmor radius for magnetic field cases
        if np.any(self.B != 0):
            B_mag = np.linalg.norm(self.B)
            v_perp_initial = np.linalg.norm(np.cross(self.v[0], self.B)) / B_mag
            larmor_radius = self.m * v_perp_initial / (abs(self.q) * B_mag)
            cyclotron_freq = abs(self.q) * B_mag / self.m
            
            results["larmor_radius"] = larmor_radius
            results["cyclotron_frequency"] = cyclotron_freq
        
        # Calculate drift velocity for crossed E-B fields
        if np.any(self.E != 0) and np.any(self.B != 0):
            B_squared = np.dot(self.B, self.B)
            if B_squared > 0:  # Avoid division by zero
                drift_velocity = np.cross(self.E, self.B) / B_squared
                results["drift_velocity"] = drift_velocity
        
        return results
    
    def plot_trajectory_2d(self, plane='xy', title=None, filename=None):
        """
        Plot the 2D projection of the particle trajectory.
        
        Parameters:
        - plane: which plane to show ('xy', 'xz', or 'yz')
        - title: custom title for the plot
        - filename: if provided, save to this filename
        """
        # Set up axes based on selected plane
        planes = {
            'xy': (0, 1, 'x', 'y'),
            'xz': (0, 2, 'x', 'z'),
            'yz': (1, 2, 'y', 'z')
        }
        
        if plane not in planes:
            raise ValueError(f"Invalid plane '{plane}'. Choose from 'xy', 'xz', or 'yz'")
            
        idx1, idx2, label1, label2 = planes[plane]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.plot(self.r[:, idx1], self.r[:, idx2], 'b-', lw=1.5)
        plt.plot(self.r[0, idx1], self.r[0, idx2], 'go', label='Start')
        plt.plot(self.r[-1, idx1], self.r[-1, idx2], 'ro', label='End')
        
        # Add field information to the plot
        if np.any(self.B != 0):
            B_str = f"B = [{self.B[0]:.1f}, {self.B[1]:.1f}, {self.B[2]:.1f}]"
        else:
            B_str = "B = 0"
            
        if np.any(self.E != 0):
            E_str = f"E = [{self.E[0]:.1f}, {self.E[1]:.1f}, {self.E[2]:.1f}]"
        else:
            E_str = "E = 0"
            
        field_text = f"{self.field_type} Field\n{E_str}\n{B_str}"
        plt.text(0.02, 0.98, field_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set labels and title
        plt.xlabel(f'{label1} position (m)')
        plt.ylabel(f'{label2} position (m)')
        
        if title is None:
            title = f'Charged Particle Motion: {self.field_type} Field ({plane}-plane)'
        plt.title(title)
        
        plt.grid(True)
        plt.legend()
        
        # Save the figure if filename is provided
        if filename:
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 2D plot to {save_path}")
            plt.close()
            return save_path
        
        return plt.gcf()
    
    def plot_trajectory_3d(self, title=None, filename=None):
        """
        Plot the full 3D trajectory.
        
        Parameters:
        - title: custom title for the plot
        - filename: if provided, save to this filename
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(self.r[:, 0], self.r[:, 1], self.r[:, 2], 'b-', lw=1.5)
        ax.plot([self.r[0, 0]], [self.r[0, 1]], [self.r[0, 2]], 'go', label='Start')
        ax.plot([self.r[-1, 0]], [self.r[-1, 1]], [self.r[-1, 2]], 'ro', label='End')
        
        # Set labels
        ax.set_xlabel('x position (m)')
        ax.set_ylabel('y position (m)')
        ax.set_zlabel('z position (m)')
        
        # Add field information
        if np.any(self.B != 0):
            B_str = f"B = [{self.B[0]:.1f}, {self.B[1]:.1f}, {self.B[2]:.1f}]"
        else:
            B_str = "B = 0"
            
        if np.any(self.E != 0):
            E_str = f"E = [{self.E[0]:.1f}, {self.E[1]:.1f}, {self.E[2]:.1f}]"
        else:
            E_str = "E = 0"
            
        field_text = f"{self.field_type} Field\n{E_str}\n{B_str}"
        ax.text2D(0.02, 0.98, field_text, transform=ax.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set title and add a grid
        if title is None:
            title = f'3D Trajectory: {self.field_type} Field'
        ax.set_title(title)
        
        ax.grid(True)
        ax.legend()
        
        # Adjust the viewing angle
        ax.view_init(elev=20, azim=35)
        
        # Save the figure if filename is provided
        if filename:
            save_path = os.path.join(img_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {save_path}")
            plt.close()
            return save_path
        
        return fig

# Function to run each scenario and generate visualizations
def run_and_visualize_scenario(scenario_name, params):
    print(f"Running scenario: {scenario_name}")
    
    sim = LorentzForceSimulator(q=params['q'], m=params['m'], dt=params['dt'], tmax=params['tmax'])
    sim.set_initial_conditions(params['r0'], params['v0'])
    
    # Set fields based on scenario type
    if scenario_name == "Uniform B Field":
        sim.set_uniform_B_field(params['B'])
    elif scenario_name == "Uniform E Field":
        sim.set_uniform_E_field(params['E'])
    elif scenario_name == "Parallel E and B Fields":
        sim.set_uniform_EB_fields(params['E'], params['B'])
    elif scenario_name == "Crossed E and B Fields":
        sim.set_uniform_EB_fields(params['E'], params['B'])
    
    # Run the simulation
    sim.run_simulation()
    
    # Calculate relevant parameters
    params_result = sim.calculate_parameters()
    print(f"Calculated parameters for {scenario_name}:")
    for key, value in params_result.items():
        if key != 'fields':
            print(f"  {key}: {value}")
        else:
            print(f"  {key}:")
            for field_key, field_value in value.items():
                print(f"    {field_key}: {field_value}")
    
    # Generate plots
    filename_base = scenario_name.lower().replace(" ", "_")
    
    # 2D plots for each plane
    xy_plot = sim.plot_trajectory_2d(plane='xy', filename=f"{filename_base}_xy.png")
    xz_plot = sim.plot_trajectory_2d(plane='xz', filename=f"{filename_base}_xz.png")
    yz_plot = sim.plot_trajectory_2d(plane='yz', filename=f"{filename_base}_yz.png")
    
    # 3D plot
    plot_3d = sim.plot_trajectory_3d(filename=f"{filename_base}_3d.png")
    
    # Return the simulator and results for further analysis
    return {
        "simulator": sim,
        "parameters": params_result,
        "plots": {
            "xy": xy_plot,
            "xz": xz_plot,
            "yz": yz_plot,
            "3d": plot_3d
        }
    }

# Main function to run all scenarios
def main():
    print("=" * 50)
    print("Lorentz Force Simulator")
    print("=" * 50)
    
    # Define common parameters
    base_params = {
        'q': 1.0,  # charge in C
        'm': 1.0,  # mass in kg
        'dt': 0.01,  # time step in s
        'tmax': 10.0,  # max simulation time in s
    }
    
    # Scenario 1: Uniform Magnetic Field (circular motion)
    uniform_B_params = base_params.copy()
    uniform_B_params.update({
        'r0': [0.0, 0.0, 0.0],  # initial position
        'v0': [0.0, 1.0, 0.2],  # initial velocity
        'B': [0.0, 0.0, 1.0]    # magnetic field along z-axis
    })
    
    # Scenario 2: Uniform Electric Field (accelerated motion)
    uniform_E_params = base_params.copy()
    uniform_E_params.update({
        'r0': [0.0, 0.0, 0.0],
        'v0': [0.1, 0.1, 0.1],
        'E': [1.0, 0.0, 0.0]    # electric field along x-axis
    })
    
    # Scenario 3: Parallel E and B Fields (helical motion)
    parallel_EB_params = base_params.copy()
    parallel_EB_params.update({
        'r0': [0.0, 0.0, 0.0],
        'v0': [0.0, 1.0, 0.1],
        'E': [0.0, 0.0, 0.5],    # electric field along z-axis
        'B': [0.0, 0.0, 1.0]     # magnetic field along z-axis
    })
    
    # Scenario 4: Crossed E and B Fields (drift motion)
    crossed_EB_params = base_params.copy()
    crossed_EB_params.update({
        'r0': [0.0, 0.0, 0.0],
        'v0': [0.0, 0.0, 0.5],
        'E': [1.0, 0.0, 0.0],    # electric field along x-axis
        'B': [0.0, 1.0, 0.0]     # magnetic field along y-axis
    })
    
    # Run all scenarios
    scenarios = {
        "Uniform B Field": uniform_B_params,
        "Uniform E Field": uniform_E_params,
        "Parallel E and B Fields": parallel_EB_params,
        "Crossed E and B Fields": crossed_EB_params
    }
    
    results = {}
    for name, params in scenarios.items():
        print("\n" + "-" * 50)
        results[name] = run_and_visualize_scenario(name, params)
        print("-" * 50)
    
    print("\nAll simulations completed successfully!")
    print(f"Visualization images saved to: {img_dir}")
    
    return results

if __name__ == "__main__":
    main() 