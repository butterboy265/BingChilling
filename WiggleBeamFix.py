import aframe as af
import numpy as np
import os
import csdl_alpha as csdl
from modopt import CSDLAlphaProblem
from modopt import SLSQP
import pyvista as pv  # Use PyVista for visualization

# Set the working directory to the specified path
working_directory = r'C:\\Users\\nssv1\\Desktop\\MS Research'
os.chdir(working_directory)

# Clear the command window
os.system('cls' if os.name == 'nt' else 'clear')

recorder = csdl.Recorder(inline=True)
recorder.start()

# Initialization parameters
n = 21  # Number of nodes along the beam
time_steps = 100  # Number of time steps
dt = 0.01  # Time step size (s)

# Mesh and material properties
mesh = np.zeros((n, 3))  # Initial mesh in numpy
mesh[:, 1] = np.linspace(0, 10, n)  # Define beam geometry along the y-axis
mesh = csdl.Variable(value=mesh)
mesh0 = mesh
#print(mesh.value)

aluminum = af.Material(E=70e9, G=26e9, density=2700)

radius = csdl.Variable(value=np.ones(n - 1) * 0.5)  # Beam radius
radius.set_as_design_variable(lower=0.1, scaler=1E1)
thickness = csdl.Variable(value=np.ones(n - 1) * 0.001)  # Beam thickness
cs = af.CSTube(radius=radius, thickness=thickness)


# Function to define a time-varying load
def time_varying_load(t,n):
    loads = csdl.Variable(value=np.zeros((n, 6)))
    load_magnitude = 20000 * np.sin(2*np.pi*t)  # Example sinusoidal load
    #print(load_magnitude)
    loads = loads.set(csdl.slice[:,2],load_magnitude)
    return loads

# Array to store displacements over time
all_displacements = csdl.Variable(value = np.zeros((time_steps, n, 3)))

# Time-stepping loop to compute displacements
for t_step in range(time_steps):
    t = t_step * dt
    
    # Create beam and frame
    beam = af.Beam(name='beam_1', mesh=mesh, material=aluminum, cs=cs)
    beam.fix(0)

    frame = af.Frame()
    frame.add_beam(beam)

    # Gravity acceleration applied in z-direction
    acc = csdl.Variable(value=np.array([0, 0, -9.81, 0, 0, 0]))
    frame.add_acc(acc)
    # Update the load based on the current time
    beam.add_load(time_varying_load(t,n))  # Apply time-varying forces to the beam
    
    # Solve the frame to get the new displacements
    frame.solve()

    # Extract displacements and store them
    disp = frame.displacement['beam_1']
    all_displacements = all_displacements.set(csdl.slice[t_step,:,:], disp)  # Store current displacements
  

all_displacements2 = all_displacements[:,1:,:]

norm_displacements = csdl.norm(all_displacements2, axes=(2,))  # axis=2 refers to the x, y, z components
max_time_displacement = csdl.maximum(norm_displacements, axes=(0,))
max_node_displacement = csdl.maximum(max_time_displacement)

#print(norm_displacements.value)
max_node_displacement.set_as_constraint(upper = 0.5, scaler = 1)


# Objective: minimize mass
mass = frame.mass
mass.set_as_objective(scaler=1)

print(af.__file__)
sol = frame._soliman()
print(sol[0].value)
print(sol[1].value)
recorder.stop()

# Run the optimization
'''sim = csdl.experimental.PySimulator(recorder)      #Pysimulator
#sim = csdl.experimental.JaxSimulator(recorder)      #JAXsimulator
prob = CSDLAlphaProblem(problem_name='single_beam', simulator=sim)
optimizer = SLSQP(prob, solver_options={'maxiter': 300, 'ftol': 1e-6, 'disp': True})
optimizer.solve()
optimizer.print_results()
#recorder.execute               only for JAX

# Output the optimized values
print("Optimized Radius:", radius.value)
print("Maximum displacement:", max_node_displacement.value)
print("Beam Mass", mass.value)'''

all_displacements = all_displacements.value
print(all_displacements.shape)
print(all_displacements[25,:,:])


#---------------------------PyVista-stuff--------------------------------

# Initialize the plotter for animation
'''plotter = pv.Plotter()

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
plotter.open_movie("beam_deformation_animation.mp4", framerate=10)

# Loop over all time steps to create the animation
for i in range(len(all_displacements)):
    # Clear the plotter to redraw
    plotter.clear()

    # Add the undeformed beam 
    add_cylinders(plotter, mesh0.value, radius.value, color="blue", label="Undeformed Beam")

    # Calculate the displaced points for the current timestep
    displaced_points = mesh0.value + all_displacements[i,:,:]  # Deformed mesh for the current timestep
    
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
plotter.show()'''
