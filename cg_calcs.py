import trimesh
import numpy as np
from vedo import Box, Mesh, show, Plotter, Point, Arrow

# List of components [x,y,z] - Mass (kg) 
# Green is moving / Red is Stationary / Fuel Tanks are Yellow
# rotation : [x_rot, y_rot, z_rot] in degrees}

components = [
    {'name': 'battery1', 'file': 'STL_Parts\Battery_8S6P_V1.STL', 'mass': 5.9, 'color': 'green', 'offset': [-100, 0, -250],'rotation': [90, 0, 0]},
    {'name': 'battery2', 'file': 'STL_Parts\Battery_8S6P_V1.STL', 'mass': 5.9, 'color': 'green', 'offset': [0, 200, -250],'rotation': [90, 0, 0]},
    {'name': 'computer', 'file': 'STL_Parts\Image_Processing_Computer_IPC_7000.stl', 'mass': 5.8, 'color': 'green', 'offset': [100, -300, -250],
        'rotation': [0, 0, 0]},
    {'name': 'PCDU', 'file': 'STL_Parts\Colossus_PCDU.stl', 'mass': 7.7, 'color': 'green', 'offset': [0, -250, 0],'rotation': [90, 0, 0]},
    # {'name': 'electric_thruster', 'file': 'STL_Parts\Electric_Thruster_V1.STL', 'mass': 1, 'color': 'red', 'offset': [0, 0, 0],
    #     'rotation': [0, 0, 0]},
    # {'name': 'solar_panels_stowed_1', 'file': 'STL_Parts\Sparkwing_36V_600x700mm_3Panel_SA_Stowed.STL', 'mass': 12.25, 'color': 'red', 
    #     'offset': [0, 0, 0],'rotation': [0, 0, 0]},
    # {'name': 'solar_panels_stowed_2', 'file': 'STL_Parts\Sparkwing_36V_600x700mm_3Panel_SA_Stowed.STL', 'mass': 12.25, 'color': 'red', 
    #     'offset': [0, 0, 0],'rotation': [0, 0, 0]},
    # {'name': 'gimbal_1', 'file': 'STL_Parts\Mood_SADA_Gimbal_V1.STL', 'mass': 1.5, 'color': 'red', 'offset': [0, 0, 0],'rotation': [0, 0, 0]},
    # {'name': 'gimbal_2', 'file': 'STL_Parts\Mood_SADA_Gimbal_V1.STL', 'mass': 1.5, 'color': 'red', 'offset': [0, 0, 0],'rotation': [0, 0, 0]},
]

# Envelope size - Sat Bus
envelope_size = np.array([711.2, 965.2, 609.6]) # mm - (28"x38"x24")
envelope = Box(pos=[0, 0, 0], size=envelope_size, c='blue5', alpha=0.2).wireframe(False)

# For CG calculation (Initialization)
total_mass = 0.0
weighted_centroids = []
meshes = [envelope]

# Mesh each component (trimesh and vedo)
for comp in components:
    mesh_tm = trimesh.load_mesh(comp['file'])
    rotation_center = mesh_tm.centroid.copy() # Make a copy of the centriod location - may not need

    # Rotate each componet
    rx, ry, rz = comp.get('rotation', [0, 0, 0])
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0], point=rotation_center))
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0], point=rotation_center))
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1], point=rotation_center))

    # Translate 
    mesh_tm.apply_translation(comp['offset'])
    centroid = mesh_tm.centroid

    mesh_vd = Mesh(mesh_tm, c=comp['color'], alpha=0.4).lighting('plastic')
    meshes.append(mesh_vd)  

    mass = comp['mass']
    weighted_centroids.append(mass * centroid)
    total_mass += mass

# Compute center of gravity (CG)
envelope_center = envelope.center_of_mass() # Finding the geometric center of mass for the envelope
CG = sum(weighted_centroids) / total_mass
print(f"Center of Gravity: {CG}")
print(f"Geometric Center: {envelope_center}")
#print(weighted_centroids)

# Add CG point & Geo Center to scene
cg_marker = Point(pos=CG, r=12, c='black')
geo_marker = Point(pos=envelope_center, r=12, c='red')
meshes.append([cg_marker,geo_marker])

# Make a vector for CG
arrow = Arrow(start_pt=envelope_center, end_pt=CG, c='red', s=0.2, alpha=1)
meshes.append(arrow)

# Display
plotter = Plotter(title='Satellite Packing', axes=1, bg='white')
plotter.show(*meshes, viewup='z', interactive=True)

