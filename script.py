import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m³/kg·s²)

# Celestial body data
bodies = {
    'Earth': {
        'mass': 5.972e24,  # kg
        'radius': 6.371e6,  # m
        'color': 'blue',
        'orbit_velocity': 29.78e3  # m/s
    },
    'Mars': {
        'mass': 6.39e23,
        'radius': 3.389e6,
        'color': 'red',
        'orbit_velocity': 24.077e3
    },
    'Jupiter': {
        'mass': 1.898e27,
        'radius': 6.9911e7,
        'color': 'orange',
        'orbit_velocity': 13.07e3
    }
}

def calculate_cosmic_velocities(mass, radius, orbit_velocity, altitudes):
    """Calculate cosmic velocities at different altitudes."""
    v1 = np.sqrt(G * mass / (radius + altitudes))  # First cosmic velocity
    v2 = v1 * np.sqrt(2)  # Second cosmic velocity
    v3 = np.sqrt(v2**2 + orbit_velocity**2)  # Third cosmic velocity
    return v1, v2, v3

# Generate altitude points (0 to 1000 km)
altitudes = np.linspace(0, 1000000, 1000)

# Create plots
plt.figure(figsize=(15, 10))

# Plot 1: Cosmic velocities vs altitude for each body
plt.subplot(1, 2, 1)
for body, data in bodies.items():
    v1, v2, v3 = calculate_cosmic_velocities(
        data['mass'], data['radius'], data['orbit_velocity'], altitudes
    )
    
    plt.plot(altitudes/1000, v1/1000, '--', 
             color=data['color'], label=f'{body} (v₁)')
    plt.plot(altitudes/1000, v2/1000, '-', 
             color=data['color'], label=f'{body} (v₂)')

plt.xlabel('Altitude (km)')
plt.ylabel('Velocity (km/s)')
plt.title('Cosmic Velocities vs Altitude')
plt.grid(True)
plt.legend()

# Plot 2: Comparison of escape velocities at surface
plt.subplot(1, 2, 2)
bodies_list = list(bodies.keys())
v1_surface = []
v2_surface = []
v3_surface = []

for body, data in bodies.items():
    v1, v2, v3 = calculate_cosmic_velocities(
        data['mass'], data['radius'], data['orbit_velocity'], np.array([0])
    )
    v1_surface.append(v1[0]/1000)
    v2_surface.append(v2[0]/1000)
    v3_surface.append(v3[0]/1000)

x = np.arange(len(bodies_list))
width = 0.25

plt.bar(x - width, v1_surface, width, label='First Cosmic Velocity',
        color=['blue', 'red', 'orange'], alpha=0.5)
plt.bar(x, v2_surface, width, label='Second Cosmic Velocity',
        color=['blue', 'red', 'orange'], alpha=0.7)
plt.bar(x + width, v3_surface, width, label='Third Cosmic Velocity',
        color=['blue', 'red', 'orange'], alpha=0.9)

plt.xlabel('Celestial Body')
plt.ylabel('Velocity (km/s)')
plt.title('Comparison of Cosmic Velocities at Surface')
plt.xticks(x, bodies_list)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cosmic_velocities.png')
plt.show()

# Create 3D visualization of escape trajectories
def plot_escape_trajectories():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Time points
    t = np.linspace(0, 10, 1000)
    
    for body, data in bodies.items():
        # Surface escape velocity
        v2 = np.sqrt(2 * G * data['mass'] / data['radius'])
        
        # Plot different escape trajectories
        for angle in [30, 45, 60]:
            # Convert angle to radians
            theta = np.radians(angle)
            
            # Initial velocities
            vx = v2 * np.cos(theta)
            vy = v2 * np.sin(theta)
            
            # Calculate positions
            x = vx * t
            y = vy * t - 0.5 * G * data['mass'] / data['radius']**2 * t**2
            z = np.zeros_like(t)
            
            # Plot trajectory
            ax.plot(x/1e6, y/1e6, z, 
                   label=f'{body} ({angle}°)', 
                   color=data['color'], 
                   alpha=0.6)
    
    # Plot celestial bodies
    for body, data in bodies.items():
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = data['radius'] * np.outer(np.cos(u), np.sin(v)) / 1e6
        y = data['radius'] * np.outer(np.sin(u), np.sin(v)) / 1e6
        z = data['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) / 1e6
        ax.plot_surface(x, y, z, color=data['color'], alpha=0.2)
    
    ax.set_xlabel('X (1000 km)')
    ax.set_ylabel('Y (1000 km)')
    ax.set_zlabel('Z (1000 km)')
    ax.set_title('Escape Trajectories from Different Celestial Bodies')
    plt.legend()
    plt.savefig('escape_trajectories.png')
    plt.show()

# Generate 3D visualization
plot_escape_trajectories()