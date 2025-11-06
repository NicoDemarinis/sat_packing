# import trimesh
# import numpy as np

# # Load STL files
# battery = trimesh.load_mesh('Battery_8S6P_V1.stl')
# computer = trimesh.load_mesh('Image_Processing_Computer_IPC_7000.stl')

# # Translate battery to origin, just to organize things visually
# battery.apply_translation(-battery.centroid)
# computer.apply_translation(-computer.centroid + [100, 0, 0])  # shift it right

# # Define envelope as a wireframe box
# envelope_size = np.array([500, 500, 500]) #np.array([711.2, 965.2, 609.6])  # mm (28x38x24")
# envelope = trimesh.creation.box(extents=envelope_size)
# #envelope.apply_translation(envelope_size / 2.0)  # shift so corner is at origin
# envelope.visual.face_colors = [0, 0, 255, 20]  # transparent blue

# # Create a scene in Trimesh
# scene = trimesh.Scene()
# scene.add_geometry(envelope, node_name='envelope')
# scene.add_geometry(battery, node_name='battery')
# #scene.add_geometry(computer, node_name='computer')

# # Show the scene in an interactive viewer
# scene.show()

################################################################ VEDO
import trimesh
import numpy as np
from vedo import Box, Mesh, show, Plotter

# Load STL files using trimesh
battery_trimesh = trimesh.load_mesh('Battery_8S6P_V1.stl')
computer_trimesh = trimesh.load_mesh('Image_Processing_Computer_IPC_7000.stl')

# Center the parts and apply translations
battery_trimesh.apply_translation(-battery_trimesh.centroid)
computer_trimesh.apply_translation(-computer_trimesh.centroid + [100, 0, 0])

# Convert to vedo meshes
battery = Mesh(battery_trimesh, c='red', alpha=1).lighting('plastic')
computer = Mesh(computer_trimesh, c='green', alpha=1).lighting('plastic')

# Envelope box — semi-transparent blue wireframe
envelope_size = np.array([711.2, 965.2, 609.6])  # mm
envelope = Box(pos=[0, 0, 0], size=envelope_size, c='blue5', alpha=0.2).wireframe(False)

# Set up vedo plotter and add all meshes
plotter = Plotter(title='Satellite Packing Viewer', axes=1, bg='white')
plotter += [envelope, battery, computer]
plotter.show(viewup='z', interactive=True)



