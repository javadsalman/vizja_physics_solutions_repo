import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 1.989e30     # Mass of the Sun (kg)

# Function to calculate orbital period based on Kepler's Third Law
def calculate_period(radius):
    return 2 * np.pi * np.sqrt(radius**3 / (G * M))

# Function to generate orbit points
def generate_orbit(radius, num_points=1000):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y

# Verify Kepler's Third Law for various radii
radii = np.linspace(0.5e11, 5.0e11, 20)  # Different orbital radii in meters
periods = [calculate_period(r) for r in radii]
periods_squared = [p**2 for p in periods]
radii_cubed = [r**3 for r in radii]

# Create plots
plt.figure(figsize=(16, 8))

# Plot 1: Orbital paths for different radii
plt.subplot(1, 2, 1)
for r in [0.5e11, 1.0e11, 2.0e11, 3.0e11]:
    x, y = generate_orbit(r)
    plt.plot(x, y)
plt.scatter(0, 0, color='yellow', s=200, label='Sun')
plt.axis('equal')
plt.grid(True)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Circular Orbits at Different Radii')
plt.legend(['0.5 AU', '1.0 AU', '2.0 AU', '3.0 AU', 'Sun'])

# Plot 2: T^2 vs r^3 (Kepler's Third Law)
plt.subplot(1, 2, 2)
plt.scatter(radii_cubed, periods_squared, color='blue')
plt.plot(radii_cubed, [(4*np.pi**2/(G*M))*r3 for r3 in radii_cubed], 'r--')
plt.xlabel('Radius Cubed (m³)')
plt.ylabel('Period Squared (s²)')
plt.title('Kepler\'s Third Law: T² ∝ r³')
plt.grid(True)
plt.legend(['Data Points', 'Theoretical Line'])

plt.tight_layout()
plt.savefig('keplers_third_law.png')
plt.show()

# Animation of orbital motion
def animate_orbits():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.grid(True)
    
    # Define planets with different radii
    planets = [
        {"radius": 0.5e11, "color": "gray", "size": 10},    # Mercury-like
        {"radius": 1.0e11, "color": "blue", "size": 20},    # Earth-like
        {"radius": 1.5e11, "color": "red", "size": 15},     # Mars-like
        {"radius": 2.5e11, "color": "orange", "size": 30}   # Jupiter-like
    ]
    
    # Calculate periods
    for planet in planets:
        planet["period"] = calculate_period(planet["radius"])
        planet["point"], = ax.plot([], [], 'o', color=planet["color"], 
                                   markersize=planet["size"])
        planet["orbit"], = ax.plot([], [], '-', color=planet["color"], alpha=0.3)
        x, y = generate_orbit(planet["radius"])
        planet["orbit_x"] = x
        planet["orbit_y"] = y
    
    # Plot Sun
    sun = plt.Circle((0, 0), 0.1e11, color='yellow')
    ax.add_patch(sun)
    
    # Set limits
    ax.set_xlim(-3e11, 3e11)
    ax.set_ylim(-3e11, 3e11)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Planet Orbits Following Kepler\'s Third Law')
    
    def init():
        for planet in planets:
            planet["point"].set_data([], [])
            planet["orbit"].set_data([], [])
        return [planet["point"] for planet in planets] + [planet["orbit"] for planet in planets]
    
    def animate(i):
        # Update each planet position
        for planet in planets:
            # Different angular velocity based on period
            angle = (i * 2 * np.pi / 100) % (2 * np.pi)
            x = planet["radius"] * np.cos(angle * 365 / planet["period"])
            y = planet["radius"] * np.sin(angle * 365 / planet["period"])
            planet["point"].set_data(x, y)
            planet["orbit"].set_data(planet["orbit_x"], planet["orbit_y"])
        
        return [planet["point"] for planet in planets] + [planet["orbit"] for planet in planets]
    
    ani = FuncAnimation(fig, animate, frames=100, init_func=init, blit=True)
    plt.close()  # Prevent display of the static plot
    return HTML(ani.to_jshtml())