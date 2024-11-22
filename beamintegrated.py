import aframe as af
import numpy as np
import os
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP
import pyvista as pv  # Use PyVista for visualization
from sankyintegrator import csdlIntegrator as integrate

# Clear the command window
os.system('cls' if os.name == 'nt' else 'clear')

# Set the working directory to the specified path
working_directory = r'C:\Users\nssv1\Desktop\MS Research'
os.chdir(working_directory)

recorder = csdl.Recorder(inline=True)
recorder.start()

# Initialization parameters
n = 21  # Number of nodes along the beam
totalSteps = 10
timeInt = csdl.Variable(value=np.array([0, 0.01]))

# Mesh and material properties
mesh = np.zeros((n, 3))  # Initial mesh in numpy
mesh[:, 1] = np.linspace(0, 10, n)  # Define beam geometry along the y-axis
mesh = csdl.Variable(value=mesh)
mesh0 = mesh

aluminum = af.Material(E=70e9, G=26e9, density=2700)

radius = csdl.Variable(value=np.ones(n - 1) * 0.5)  # Beam radius
radius.set_as_design_variable(lower=0.1, scaler=1E1)
thickness = csdl.Variable(value=np.ones(n - 1) * 0.001)  # Beam thickness
cs = af.CSTube(radius=radius, thickness=thickness)

beam = af.Beam(name='beam_1', mesh=mesh, material=aluminum, cs=cs)
beam.fix(0)

frame = af.Frame()
frame.add_beam(beam)

# Gravity acceleration applied in z-direction
acc = csdl.Variable(value=np.array([0, 0, -9.81, 0, 0, 0]))
frame.add_acc(acc)
# Update the load based on the current time
    
# Solve the frame to get the new displacements
#frame.solve()

sol = frame._soliman()
K = sol[0]
M = sol[1]
C = 0.04*K + 0.1*M

def beamSystem(t, y, M, K, n):

    
    # Define the force in the z-direction for each node
    F= csdl.Variable(value = np.zeros(n * 6))
    F_external = 0.01 * csdl.sin(2 * 3.14159265358979323846264 * t)
    g = -9.81
    #print(F_external)

    # Populate F with gravitational force and external force in the z-direction for each node
    for i in range(1, n):  # Skip the fixed first node

        z_index = i * 6 + 2
        F_gravity = M[z_index, z_index] * g
        F = F.set(csdl.slice[z_index], F_external + F_gravity)
    
        #print(f"Node {i}, F_external: {F_external}, F_gravity_z: {F_gravity_z}, Total F: {F_vec[z_index]}")

    #print(F[0:10]))
    
    # Extract the displacement (y0) and velocity (y1) from y as 1D slices
    y0 = y[:n * 6]      # Displacement part (first n*6 elements)
    y1 = y[n * 6:]      # Velocity part (remaining elements)

    dy0dt = y1
    dy1dt = csdl.solve_linear(M,F-csdl.matvec(C,y1) - csdl.matvec(K,y0))

    #print(t.value)
    #print(f" F_external: {F_external.value}, F_gravity_z: {F_gravity_z.value}")
    return csdl.vstack([dy0dt, dy1dt]).flatten()


y0_init = np.zeros(6*n)
y1_init = np.zeros(6*n)
initCond = np.hstack([y0_init, y1_init])
initCond = csdl.Variable(value=initCond)

t, y = integrate(beamSystem, timeInt, initCond, M, K, n, method='trapezoid', numSteps=totalSteps)

#print(y[25,:].value)
print(y.shape)

# Extract only displacement components (first three values in each group of six for each node)
displacement_data = y[:, :n * 6].value  # First n*6 elements for displacement
displacement_data = displacement_data.reshape((totalSteps + 1, n, 6))  # Reshape to (timesteps, nodes, 6)

# Select only the displacement components: x, y, z (skip velocity components)
displacement_data = displacement_data[:, :, :3]  # Shape: (numSteps + 1, n, 3)
print(displacement_data.shape)
print(displacement_data[:,-1,:])



#-------------------------------Pyvista-Cringe-------------------------------------------------------
# Initialize the plotter for animation
plotter = pv.Plotter()

# Function to add a cylinder for each beam segment
def add_cylinders(plotter, points, radius, color, label):
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        center = (start + end) / 2.0
        direction = end - start
        height = np.linalg.norm(direction)  # Length of the cylinder
        direction /= height  # Normalize the direction

        # Create a cylinder with the center, direction, and radius
        cylinder = pv.Cylinder(center=center, direction=direction, radius=radius[i], height=height)
        plotter.add_mesh(cylinder, color=color, opacity=0.6, label=label)

# Open a movie file to capture the animation frames
plotter.open_movie("beam_wiggle.mp4", framerate=10)

# Loop over all time steps to create the animation
for i in range(displacement_data.shape[0]):
    # Clear the plotter to redraw
    plotter.clear()

    # Add the undeformed beam 
    add_cylinders(plotter, mesh0.value, radius.value, color="blue", label="Undeformed Beam")

    # Calculate the displaced points for the current timestep
    displaced_points = mesh0.value + displacement_data[i]  # Deformed mesh for the current timestep
    
    # Add the deformed beam
    add_cylinders(plotter, displaced_points, radius.value, color="red", label="Deformed Beam")

    # Add the legend for identifying the original and deformed beams
    plotter.add_legend(
        labels=[("Undeformed Beam", "blue"), ("Deformed Beam", "red")],
        face='circle',  # Use "circle" as the face shape
        border=True,
        loc="upper right"  # Move the legend to the upper right
    )

    # Add the z-axis line from -3 to +3
    scale_position = [0, 0, 0]  # Starting position at origin
    scale_start = [0, 0, -3]  # Z-axis starts at -3
    scale_end = [0, 0, 3]  # Z-axis ends at +3
    plotter.add_mesh(pv.Line(scale_start, scale_end), color='black', line_width=5)

    # Add tick marks along the z-axis
    tick_values = np.linspace(-3, 3, 7)  # Create tick marks from -3 to 3 (7 ticks in total)
    tick_length = 0.2  # Length of the tick marks
    for tick in tick_values:
        # Add tick marks
        start = [scale_position[0] - tick_length / 2, scale_position[1], tick]
        end = [scale_position[0] + tick_length / 2, scale_position[1], tick]
        plotter.add_mesh(pv.Line(start, end), color='black', line_width=2)
        
        # Add tick labels
        plotter.add_point_labels(
            points=[[scale_position[0] + tick_length, scale_position[1], tick]], 
            labels=[f"{tick:.1f}"], 
            font_size=12, 
            text_color="black", 
            point_color=None, 
            point_size=0
        )

    # Add a bounding box around the model
    plotter.add_bounding_box(color="lightgrey", opacity=0.7)

    # Show axes and set the view to the XY plane
    plotter.show_axes()
    plotter.view_isometric()
    #plotter.view_yz()
    # Capture the current frame for the animation
    plotter.write_frame()

# Close the movie file to finalize the animation
plotter.close()

# Optionally, show the plot after creating the animation
plotter.show()
