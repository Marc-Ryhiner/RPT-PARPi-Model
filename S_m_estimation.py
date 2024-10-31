import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid OpenGL errors

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


np.random.seed(17)

# Cylinder dimensions
cylinder_radius = 0.343  # cm
cylinder_height = 0.271  # cm
num_spheres = 16407  # Total number of spherical cells
cell_radius = 0.0006  # cm (6 micrometers for better spacing)

# Gamma source (140 keV, 1 MBq)
gamma_energy_keV = 140  # keV
gamma_activity_MBq = 1  # MBq

# Linear attenuation coefficient for water at 140 keV (in cm^-1)
attenuation_coeff_water = 0.153  # cm^-1 (based on NIST data for water at 140 keV)

# Number of particles to simulate
N_Particles = 1e9

# Generate uniform spherical cells inside the cylinder
def generate_uniform_points_in_cylinder(num_points, radius, height):
    z_values = np.linspace(-height / 2, height / 2, int(np.sqrt(num_points)))
    theta_values = np.linspace(0, 2 * np.pi, int(np.sqrt(num_points)), endpoint=False)

    spherical_cells = []
    for z in z_values:
        for theta in theta_values:
            r = np.sqrt(np.random.uniform(0, radius ** 2))  # Uniformly distributed radius
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            spherical_cells.append((x, y, z))

    return np.array(spherical_cells)

# Generate the cells
spherical_cells = generate_uniform_points_in_cylinder(num_spheres, cylinder_radius, cylinder_height)


# Function to calculate S-values (Gy/Bq·s) with water attenuation
def calculate_s_values_with_water(cells, gamma_activity_Bq, attenuation_coeff, N_Particles):
    """
    Calculate the S-values (dose per unit activity) for each spherical cell.
    This considers water attenuation based on the distance of cells from the source.
    """
    # Calculate the distance of each cell from the center (gamma source location)
    distance = np.linalg.norm(cells, axis=1)  # Distance from the source (0,0,0)

    # Simplified dose calculation using inverse square law
    dose_without_attenuation = gamma_activity_Bq / (4 * np.pi * distance**2)

    # Apply water attenuation to the dose
    dose_with_attenuation = dose_without_attenuation * np.exp(-attenuation_coeff * distance)

    # Calculate S-value map (dose per particle per unit activity)
    s_value_map = dose_with_attenuation / N_Particles  # Gy/Bq·s
    
    return s_value_map


# Calculate S-values (dose values) for each spherical cell
s_values = calculate_s_values_with_water(spherical_cells, gamma_activity_MBq, attenuation_coeff_water, N_Particles)

# Calculate the mean S-value and uncertainties
s_value_source = np.mean(s_values)  # Gy/Bq·s
s_value_source_formatted = "%.2E" % s_value_source  # Format in scientific notation

# Uncertainty (using standard deviation)
uncertainty = np.std(s_values) / np.mean(s_values) * 100  # Percentage uncertainty
unc_source_formatted = "%.2E" % uncertainty  # Format in scientific notation

# Output the final S-value for the gamma source
print(f"Mean S-value (in Gy/Bq·s) for the source: {s_value_source_formatted}")
print(f"Uncertainty in the S-value (%): {unc_source_formatted}")

# Create a text file to save S-values, mean S-value, and uncertainty
with open("s_value_results.txt", "w") as file:
    file.write("S-Value Results\n")
    file.write("=================\n")
    file.write(f"Mean S-value (Gy/Bq·s): {s_value_source_formatted}\n")
    file.write(f"Uncertainty in the S-value (%): {unc_source_formatted}\n\n")
    file.write("Individual S-values:\n")
    np.savetxt(file, s_values, header='S-values (Gy/Bq·s)', delimiter=',')

# ---------- Visualization of Geometry and Particle Emission Animation -----------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot spherical cells inside the cylinder
ax.scatter(spherical_cells[:, 0], spherical_cells[:, 1], spherical_cells[:, 2], c='b', marker='o', s=2, alpha=0.5)
ax.set_xlabel('X axis (cm)')
ax.set_ylabel('Y axis (cm)')
ax.set_zlabel('Z axis (cm)')
ax.set_title('Spherical Cells in Cylinder')

# Plot the source at the center in red
ax.scatter(0, 0, 0, c='r', marker='o', s=100, label='Gamma Source')  # Increased size for visibility

# Set limits to visualize the entire geometry
ax.set_xlim([-cylinder_radius, cylinder_radius])
ax.set_ylim([-cylinder_radius, cylinder_radius])
ax.set_zlim([-cylinder_height/2, cylinder_height/2])

# Animation: Simulate particle release
def update(frame):
    ax.view_init(elev=20., azim=frame)


ani = animation.FuncAnimation(fig, update, frames=range(0, 360, 2), interval=50)
ani.save("gamma_particle_release.gif", writer='imagemagick')
#ani.save("gamma_particle_release.png", writer='imagemagick')


plt.close()  # Close plot to avoid display issues

# Plot dose (S-values) distribution as a histogram including all values
plt.hist(s_values * 1e3, bins=100, color='blue', alpha=0.5, label='All S-values (including zeros)')  # Convert to mGy for plotting
plt.title('S-Value (Dose per Activity) Distribution with Water Attenuation')
plt.xlabel('S-value (mGy/MBq)')
plt.ylabel('Number of cells')
plt.legend()
plt.savefig("s_value_distribution_all.png")  # Save the plot to a file
plt.close()  # Close the plot to avoid display issues

# Plot dose (S-values) distribution as a histogram focusing on non-zero values
plt.figure(figsize=(10, 6))
plt.hist(s_values[(s_values > 0)], bins=100, color='blue', alpha=0.7, range=(1e-11, 5e-7))  # Focus on non-zero values
plt.title('S-Value (Dose per Activity) Distribution with Water Attenuation')
plt.xlabel('S-value (Gy/Bq·s)')
plt.ylabel('Number of cells')
plt.xlim(1e-11, 5e-7)  # Set x-limits
plt.yscale('log')  # Optional: use a logarithmic scale for better visibility
plt.grid(True)
plt.savefig("s_value_distribution.png")  # Save the plot to a file
plt.close()  # Close the plot to avoid display issues

